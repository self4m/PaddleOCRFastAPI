import asyncio
import base64
import gc
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from fastapi import UploadFile, File

# 字体文件，两种字体二选一
# os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = Path("fonts/PingFang-SC-Regular.ttf").resolve().as_posix()
os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = Path("fonts/simfang.ttf").resolve().as_posix()

from paddleocr import PaddleOCR
import fastapi
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

font_path = Path("./fonts/PingFang-SC-Regular.ttf").resolve()
LOCAL_FONT_FILE_PATH = str(font_path)

MAX_CONCURRENT_OCR = 4
ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR)
app=fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OCR 引擎
pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")

@app.post("/ocr", response_class=JSONResponse, summary="识别图片或PDF中的文字")
async def extract_text(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()

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
            result = await loop.run_in_executor(thread_pool, partial(pipeline.predict, ocr_file_path))

            temp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            text_list = []
            image_list = []
            num = 1
            for res in result:
                rec_texts = res.get('rec_texts', [])
                text_list.extend(rec_texts)
                res.save_to_img(os.path.join(temp_dir, f"{num}.png"))
                num = num + 1
            file_names = sorted(os.listdir(temp_dir))
            for file_name in file_names:
                if file_name.endswith(".png"):
                    file_path = os.path.join(temp_dir, file_name)
                    with open(file_path, "rb") as f:
                        img_bytes = f.read()
                        img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
                        image_list.append(img_b64)

            detail_time = time.time() - start_time
            return JSONResponse({"result": text_list, "detail_time": detail_time, "image_list": image_list},status_code=200)
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return JSONResponse(content={"error": f"OCR识别失败: {e}"}, status_code=500)
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)
            gc.collect()
