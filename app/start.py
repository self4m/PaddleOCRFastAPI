from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR
from dotenv import load_dotenv
import numpy as np
import fitz  # PyMuPDF
import os
import cv2
import tempfile
from PIL import Image

# 初始化 FastAPI 应用
app = FastAPI(title="百度飞桨 PaddleOCR 文字识别")

# 加载环境变量
load_dotenv()

# 获取模型路径
det_model_dir = os.getenv("DET_MODEL_DIR")
rec_model_dir = os.getenv("REC_MODEL_DIR")
cls_model_dir = os.getenv("CLS_MODEL_DIR")

# 校验模型路径存在性
if not all([det_model_dir, rec_model_dir, cls_model_dir]):
    raise RuntimeError("环境变量未正确设置：请检查 DET_MODEL_DIR、REC_MODEL_DIR、CLS_MODEL_DIR")

# 初始化 OCR 引擎（仅初始化一次）
ocr = PaddleOCR(
    det_model_dir=det_model_dir,
    rec_model_dir=rec_model_dir,
    cls_model_dir=cls_model_dir,
    lang="ch",
    use_angle_cls=True,
    det_db_thresh=0.2,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=2.0,
    drop_score=0.3
)

@app.post("/ocr_image", response_class=PlainTextResponse, summary="提取图片文字")
async def extract_text_from_image(file: UploadFile = File(...)) -> str:
    """
    上传图片并返回提取的文字内容（每行一个文本）

    参数:
    - file: 上传的图片文件（jpg/png）

    返回:
    - str: 识别到的文本内容（以换行分隔）
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件（jpg/png）")

    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法解码图像")

        # 提高对比度
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

        result = ocr.ocr(enhanced_image)
        texts = [line[1][0] for line in result[0] if line and line[1]]

        return "\n".join(texts) if texts else "未识别到文字"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/ocr_pdf", response_class=PlainTextResponse, summary="提取 PDF 中文字")
async def extract_text_from_pdf(file: UploadFile = File(...)) -> str:
    """
    上传 PDF 并返回识别的文字内容（按页输出）

    参数:
    - file: 上传的 PDF 文件

    返回:
    - str: 所有页面识别到的文字（以换行分隔）
    """
    if not file.content_type or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传有效的 PDF 文件")

    try:
        # 保存临时 PDF 文件
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(contents)
            tmp_pdf_path = tmp_pdf.name

        # 打开 PDF 并逐页转换为图片
        text_result = []
        with fitz.open(tmp_pdf_path) as pdf:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # 提高对比度
                enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

                # OCR 识别
                result = ocr.ocr(enhanced_image, cls=True)
                page_texts = [line[1][0] for line in result[0] if line and line[1]]
                text_result.append(f"第 {page_index + 1} 页：\n" + "\n".join(page_texts))

        os.remove(tmp_pdf_path)
        return "\n\n".join(text_result) if text_result else "未识别到文字"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
