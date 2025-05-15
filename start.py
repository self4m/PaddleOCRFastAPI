import gc
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR
from pydantic.v1 import BaseSettings


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
ocr_light = PaddleOCR(
    det_model_dir=settings.det_model_dir,
    rec_model_dir=settings.rec_model_dir,
    cls_model_dir=settings.cls_model_dir,
    lang="ch",
    use_angle_cls=True
)

ocr_full = PaddleOCR(
    det_model_dir=settings.det_model_dir_server,
    rec_model_dir=settings.rec_model_dir_server,
    cls_model_dir=settings.cls_model_dir,
    lang="ch",
    use_angle_cls=True
)


@asynccontextmanager
async def image_processing_context(file: UploadFile):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传有效的图片文件（jpg/png）")

        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像")

        enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        yield enhanced
    finally:
        gc.collect()


@asynccontextmanager
async def pdf_processing_context(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传有效的 PDF 文件")

    contents = await file.read()

    tmp_dir = tempfile.mkdtemp()
    tmp_pdf_path = os.path.join(tmp_dir, "temp.pdf")

    try:
        with open(tmp_pdf_path, "wb") as f:
            f.write(contents)

        images = []
        with fitz.open(tmp_pdf_path) as pdf:
            for page in pdf:
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
                images.append(enhanced)

        yield images
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # 彻底删除整个临时目录及文件
        gc.collect()


def process_ocr_result(result):
    texts = [line[1][0] for line in result[0] if line and line[1]]
    return "\n".join(texts) if texts else "未识别到文字"


@app.post("/ocr_image_light", response_class=PlainTextResponse, summary="提取图片文字——轻量模型")
async def extract_text_from_image_light(file: UploadFile = File(...)):
    async with image_processing_context(file) as image:
        try:
            result = ocr_light.ocr(image, det=True, cls=False)
            return process_ocr_result(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/ocr_image_full", response_class=PlainTextResponse, summary="提取图片文字——全量模型")
async def extract_text_from_image_full(file: UploadFile = File(...)):
    async with image_processing_context(file) as image:
        try:
            result = ocr_full.ocr(image, det=True, cls=False)
            return process_ocr_result(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/ocr_pdf_light", response_class=PlainTextResponse, summary="提取 PDF 中文字（轻量模型）")
async def extract_text_from_pdf_light(file: UploadFile = File(...)):
    async with pdf_processing_context(file) as images:
        try:
            texts = []
            for image in images:
                result = ocr_light.ocr(image, det=True, cls=False)
                page_text = process_ocr_result(result)
                texts.append(page_text)
            return "\n".join(texts) if texts else "未识别到文字"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/ocr_pdf_full", response_class=PlainTextResponse, summary="提取 PDF 中文字（全量模型）")
async def extract_text_from_pdf_full(file: UploadFile = File(...)):
    async with pdf_processing_context(file) as images:
        try:
            texts = []
            for image in images:
                result = ocr_full.ocr(image, det=True, cls=False)
                page_text = process_ocr_result(result)
                texts.append(page_text)
            return "\n".join(texts) if texts else "未识别到文字"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("start:app", host="127.0.0.1", port=8888, reload=True)
