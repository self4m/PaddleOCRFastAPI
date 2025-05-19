import base64
import gc
import io
import time

import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
    texts = [line[1][0] for line in ocr_result[0]]
    result = " ".join(texts).replace(" ", "")
    print(result)
    detail_time = time.time() - start_time
    return JSONResponse(content={"result": result, "detail_time": detail_time}, status_code=200)


# 按行输出最终结果
def return_final_result(ocr_result, start_time):
    result = []
    for line in ocr_result[0]:
        texts = line[1][0]
        result.append(texts)
        print(texts)
    detail_time = time.time() - start_time
    return JSONResponse(content={"result": result, "detail_time": detail_time}, status_code=200)


# 输出坐标、结果、准确率
def return_result(ocr_result, start_time):
    result = []
    for line in ocr_result[0]:
        result.append(line)
        print(line)
    detail_time = time.time() - start_time
    return JSONResponse(content={"result": result, "detail_time": detail_time}, status_code=200)


# 输出坐标、结果、准确率 + 标注图像的识别结果
def return_result_with_result_photo(ocr_result, img, start_time):
    result = []
    for line in ocr_result[0]:
        result.append(line)
        print(line)

    # 标注图像的识别结果
    ocr_result = ocr_result[0]
    # image = Image.open(img).convert('RGB')
    boxes = [line[0] for line in ocr_result]
    txts = [line[1][0] for line in ocr_result]
    scores = [line[1][1] for line in ocr_result]
    im_show = draw_ocr(img, boxes, txts, scores, font_path='../model/font/simfang.ttf')
    im_show = Image.fromarray(im_show)
    buffer = io.BytesIO()
    im_show.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    detail_time = time.time() - start_time
    return JSONResponse(content={"result": result, "image_base64": img_base64, "detail_time": detail_time},
                        status_code=200)


# 定义全局结果输出方式
def return_result_mode(ocr_result, img, result_type, start_time):
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
        return return_result_with_result_photo(ocr_result, img, start_time)


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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件（jpg/png）")
    contents = await file.read()
    try:
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "无法打开图片"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"打开图片失败：{e}"}, status_code=400)
    try:
        result = ocr_engine(model).ocr(img, det=True, cls=False)
        response = return_result_mode(result, img, result_type, start_time)
        gc.collect()
        return response
    except Exception as e:
        print(f"发生错误: {e}")
        gc.collect()
        return JSONResponse(content={"error": f"服务器内部错误: {e}"}, status_code=500)


# 主函数
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8888)
