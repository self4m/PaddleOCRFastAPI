import gc
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR
from pydantic.v1 import BaseSettings
from contextlib import asynccontextmanager


class Settings(BaseSettings):
    det_model_dir: str = "model/det/ch_PP-OCRv4_det_infer"
    rec_model_dir: str = "model/rec/ch_PP-OCRv4_rec_infer"
    cls_model_dir: str = "model/cls/ch_ppocr_mobile_v2.0_cls_infer"
    det_model_dir_server: str = "model/det/ch_PP-OCRv4_det_server_infer"
    rec_model_dir_server: str = "model/rec/ch_PP-OCRv4_rec_server_infer"


settings = Settings()

# 初始化 FastAPI 应用
app = FastAPI(title="百度飞桨 PaddleOCR 文字识别")

# 初始化 OCR 引擎
ocr = PaddleOCR(
    det_model_dir=settings.det_model_dir,
    rec_model_dir=settings.rec_model_dir,
    cls_model_dir=settings.cls_model_dir,
    lang="ch",
    use_angle_cls=True
)

ocr_server = PaddleOCR(
    det_model_dir=settings.det_model_dir_server,
    rec_model_dir=settings.rec_model_dir_server,
    cls_model_dir=settings.cls_model_dir,
    lang="ch",
    use_angle_cls=True
)


async def process_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件（jpg/png）")

    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图像")

    return cv2.convertScaleAbs(image, alpha=1.5, beta=0)


def process_ocr_result(result):
    texts = [line[1][0] for line in result[0] if line and line[1]]
    return "\n".join(texts) if texts else "未识别到文字"


@asynccontextmanager
async def image_processing_context(file):
    try:
        image = await process_image(file)
        yield image
    finally:
        gc.collect()


@app.post("/ocr_image_light", response_class=PlainTextResponse, summary="提取图片文字——超轻量")
async def extract_text_from_image_light(file: UploadFile = File(...)):
    async with image_processing_context(file) as image:
        try:
            result = ocr.ocr(image, det=True, cls=False)
            return process_ocr_result(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/ocr_image_full", response_class=PlainTextResponse, summary="提取图片文字——全量模型")
async def extract_text_from_image_full(file: UploadFile = File(...)):
    async with image_processing_context(file) as image:
        try:
            result = ocr_server.ocr(image, det=True, cls=False)
            return process_ocr_result(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("start:app", host="127.0.0.1", port=8888, reload=True)