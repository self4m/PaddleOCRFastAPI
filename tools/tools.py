import base64
import gc
import io
import os
import tempfile
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import fitz
import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from paddleocr import PaddleOCR, draw_ocr
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# 初始化 OCR 模型（单例）
ocr_light = PaddleOCR(
    det_model_dir="../model/cls/ch_PP-OCRv4_det_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)
ocr_server = PaddleOCR(
    det_model_dir="../model/cls/ch_PP-OCRv4_det_server_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_server_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)

VALID_RESULT_TYPES = {
    'return_final_result_one_line',
    'return_final_result',
    'return_result',
    'return_result_with_result_photo'
}

VALID_MODELS = {
    'ocr_light',
    'ocr_server'
}

def get_ocr_engine(model):
    if model == "ocr_light":
        return ocr_light
    elif model == "ocr_server":
        return ocr_server

# 限流信号量（最大同时运行OCR数量）
MAX_CONCURRENT_OCR = 4
ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)

# 线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR)

# PDF 处理为图像数组（同步）
def load_images_from_pdf(content: bytes) -> list:
    # 使用临时文件，使用with确保文件关闭
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(content)

        images = []
        with fitz.open(pdf_path) as pdf:
            for pg in range(pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)

        return images

# 图片/PDF 统一处理入口（同步）
def load_images_from_bytes(content: bytes, filename: str, content_type: str) -> list:
    if filename.endswith(".pdf"):
        return load_images_from_pdf(content)
    elif content_type and content_type.startswith("image/"):
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像")
        return [img]
    else:
        raise ValueError("不支持的文件类型")

# OCR同步调用封装（线程池执行）
def ocr_sync_call(ocr_engine, images):
    return ocr_engine.ocr(images, det=True, cls=False)

# 各种输出模式封装
def return_final_result_one_line(ocr_result, images, start_time):
    texts = [line[1][0] for ocr_result_idx in ocr_result for line in ocr_result_idx]
    result = "".join(texts)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_final_result(ocr_result, images, start_time):
    result = [line[1][0] for ocr_result_idx in ocr_result for line in ocr_result_idx]
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_result(ocr_result, images, start_time):
    result = [line for ocr_result_idx in ocr_result for line in ocr_result_idx]
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_result_with_result_photo(ocr_result, images, start_time):
    result = []
    image_base64_list = []
    for i in range(len(ocr_result)):
        res = ocr_result[i]
        image = images[i]
        for line in res:
            result.append(line)
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='../model/font/simfang.ttf')
        im_show = Image.fromarray(im_show)
        buffer = io.BytesIO()
        im_show.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_base64_list.append(img_base64)

    detail_time = time.time() - start_time
    return JSONResponse({
        "result": result,
        "image_base64_list": image_base64_list,
        "detail_time": detail_time
    }, status_code=200)

# 映射字典统一调度
RESULT_TYPE_FUNC_MAP = {
    'return_final_result_one_line': return_final_result_one_line,
    'return_final_result': return_final_result,
    'return_result': return_result,
    'return_result_with_result_photo': return_result_with_result_photo
}

def return_result_mode(ocr_result, images, result_type, start_time):
    if not ocr_result or not ocr_result[0]:
        detail_time = time.time() - start_time
        return JSONResponse(content={"error": "未检测到文本", "detail_time": detail_time}, status_code=400)

    return RESULT_TYPE_FUNC_MAP[result_type](ocr_result, images, start_time)

# 创建 FastAPI 实例
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 主接口（异步，带信号量限流与线程池OCR调用）
@app.post("/ocr", response_class=JSONResponse, summary="识别图片或PDF中的文字")
async def extract_text(
    file: UploadFile = File(...),
    result_type: str = Form(...),
    model: str = Form(...)
) -> JSONResponse:
    start_time = time.time()

    if result_type not in VALID_RESULT_TYPES:
        return JSONResponse(content={"error": "无效的 result_type 参数"}, status_code=400)
    if model not in VALID_MODELS:
        return JSONResponse(content={"error": "无效的 model 参数"}, status_code=400)

    try:
        contents = await file.read()
        images = await asyncio.get_running_loop().run_in_executor(None, load_images_from_bytes, contents, file.filename.lower(), file.content_type)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"文件处理失败: {e}"}, status_code=500)

    async with ocr_semaphore:
        try:
            # 使用线程池异步执行OCR，避免阻塞事件循环
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(thread_pool, ocr_sync_call, get_ocr_engine(model), images)
            return return_result_mode(result, images, result_type, start_time)
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return JSONResponse(content={"error": f"OCR识别失败: {e}"}, status_code=500)
        finally:
            gc.collect()

# 启动服务
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8888)
