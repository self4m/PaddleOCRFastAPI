import base64
import gc
import io
import os
import tempfile
import time

import cv2
import fitz
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from paddleocr import PaddleOCR, draw_ocr
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

ocr = PaddleOCR(
    det_model_dir="../model/cls/ch_PP-OCRv4_det_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)

ocr_full = PaddleOCR(
    det_model_dir="../model/cls/ch_PP-OCRv4_det_server_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_server_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)


# 一行输出最终结果
def return_final_result_one_line(ocr_result, start_time):
    texts = [line[1][0] for ocr_result_idx in ocr_result for line in ocr_result_idx]
    result = " ".join(texts).replace(" ", "")
    print(result)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)


# 按行输出最终结果
def return_final_result(ocr_result, start_time):
    result = []
    for ocr_result_idx in ocr_result:
        for line in ocr_result_idx:
            text = line[1][0]
            result.append(text)
            print(text)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)


# 输出坐标、结果、准确率
def return_result(ocr_result, start_time):
    result = []
    for ocr_result_idx in ocr_result:
        for line in ocr_result_idx:
            result.append(line)
            print(line)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "detail_time": detail_time}, status_code=200)


# 输出坐标、结果、准确率 + 标注图像的识别结果
def return_result_with_result_photo(ocr_result, pdf_path, start_time):
    result = []
    for ocr_result_idx in ocr_result:
        for line in ocr_result_idx:
            result.append(line)
            print(line)

    imgs = []
    image_base64_list = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    for ocr_result_idx in range(len(ocr_result)):
        res = ocr_result[ocr_result_idx]
        image = imgs[ocr_result_idx]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='font/simfang.ttf')
        im_show = Image.fromarray(im_show)
        buffer = io.BytesIO()
        im_show.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_base64_list.append(img_base64)
    detail_time = time.time() - start_time
    return JSONResponse({"result": result, "image_base64_list": image_base64_list, "detail_time": detail_time},
                        status_code=200)


# 定义全局结果输出方式
def return_result_mode(ocr_result, pdf_path, result_type, start_time):
    # 检查输出内容
    if not ocr_result or not ocr_result[0]:
        detail_time = time.time() - start_time
        print("未检测到文本")
        return JSONResponse(content={"error": "未检测到文本", "detail_time": detail_time}, status_code=400)

    # 一行输出最终结果
    if result_type == 'return_final_result_one_line':
        return return_final_result_one_line(ocr_result, start_time)

    # 按行输出最终结果
    elif result_type == 'return_final_result':
        return return_final_result(ocr_result, start_time)

    # 按行输出坐标、结果、准确率
    elif result_type == 'return_result':
        return return_result(ocr_result, start_time)

    # 按行输出坐标、结果、准确率 + 标注图像的识别结果
    elif result_type == 'return_result_with_result_photo':
        return return_result_with_result_photo(ocr_result, pdf_path, start_time)



def ocr_engine(model):
    if model == "ocr":
        return ocr
    elif model == "ocr_full":
        return ocr_full


VALID_RESULT_TYPES = {
    'return_final_result_one_line',
    'return_final_result',
    'return_result',
    'return_result_with_result_photo'
}

VALID_MODELS = {
    'ocr',
    'ocr_full'
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请指定前端地址，例 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr", response_class=JSONResponse, summary="提取图片文字")
async def extract_text(file: UploadFile = File(...), result_type: str = Form(''),
                       model: str = Form('')) -> JSONResponse:
    start_time = time.time()  # 记录开始时间
    if result_type not in VALID_RESULT_TYPES:
        return JSONResponse(content={"error": "无效的 result_type 参数"}, status_code=400)
    if model not in VALID_MODELS:
        return JSONResponse(content={"error": "无效的 model 参数"}, status_code=400)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传有效的 PDF 文件")
    contents = await file.read()
    temp_directory = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_directory, "input.pdf")
    try:
        with open(pdf_path, "wb") as file:
            file.write(contents)

        result = ocr_engine(model).ocr(pdf_path, det=True, cls=False)
        response = return_result_mode(result, pdf_path, result_type, start_time)
        gc.collect()
        return response
    except Exception as e:
        print(f"发生错误: {e}")
        gc.collect()
        return JSONResponse(content={"error": f"服务器内部错误: {e}"}, status_code=500)


# 主函数
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8888)
