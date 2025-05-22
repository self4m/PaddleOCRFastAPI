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
    rec_model_dir="model/rec/ch_PP-OCRv4_rec_infer",
    cls_model_dir="model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)
ocr_server = PaddleOCR(
    det_model_dir="../model/cls/ch_PP-OCRv4_det_server_infer",
    rec_model_dir="model/rec/ch_PP-OCRv4_rec_server_infer",
    cls_model_dir="model/cls/ch_ppocr_mobile_v2.0_cls_infer",
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

MAX_CONCURRENT_OCR = 4
ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR)

def ocr_sync_call(ocr_engine, ocr_file):
    return ocr_engine.ocr(ocr_file, det=True, cls=False)

def return_final_result_one_line(ocr_result, file, content, start_time):
    texts = [line[1][0] for ocr_result_idx in ocr_result for line in ocr_result_idx]
    result = "".join(texts)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_final_result(ocr_result, file, content, start_time):
    result = [line[1][0] for ocr_result_idx in ocr_result for line in ocr_result_idx]
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_result(ocr_result, file, content, start_time):
    result = [line for ocr_result_idx in ocr_result for line in ocr_result_idx]
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)

def return_result_with_result_photo(ocr_result, file, content, start_time):
    result = []
    image_base64_list = []
    imgs = []

    if file.content_type.startswith("image/"):
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        imgs.append(img)
    elif file.content_type == "application/pdf":
        with fitz.open(stream=content, filetype="pdf") as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)

    for i in range(len(ocr_result)):
        res = ocr_result[i]
        image = imgs[i]
        for line in res:
            result.append(line)
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='model/font/simfang.ttf')
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

RESULT_TYPE_FUNC_MAP = {
    'return_final_result_one_line': return_final_result_one_line,
    'return_final_result': return_final_result,
    'return_result': return_result,
    'return_result_with_result_photo': return_result_with_result_photo
}

def return_result_mode(ocr_result, file, content, result_type, start_time):
    if not ocr_result or not ocr_result[0]:
        detail_time = time.time() - start_time
        return JSONResponse(content={"error": "未检测到文本", "detail_time": detail_time}, status_code=400)
    return RESULT_TYPE_FUNC_MAP[result_type](ocr_result, file, content, start_time)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        file_content = await file.read()
        temp_directory = tempfile.mkdtemp()

        if file.content_type.startswith("image/"):
            ocr_file_path = os.path.join(temp_directory, "input.png")
        elif file.content_type == "application/pdf":
            ocr_file_path = os.path.join(temp_directory, "input.pdf")
        else:
            return JSONResponse(content={"error": "不支持的文件类型"}, status_code=400)

        with open(ocr_file_path, "wb") as f:
            f.write(file_content)

    except Exception as e:
        print(f"文件处理错误: {e}")
        gc.collect()
        return JSONResponse(content={"error": f"文件处理失败: {e}"}, status_code=500)

    async with ocr_semaphore:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(thread_pool, ocr_sync_call, get_ocr_engine(model), ocr_file_path)
            return return_result_mode(result, file, file_content, result_type, start_time)
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return JSONResponse(content={"error": f"OCR识别失败: {e}"}, status_code=500)
        finally:
            gc.collect()
