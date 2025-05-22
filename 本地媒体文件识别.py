from paddleocr import PaddleOCR

# 初始化 OCR 引擎
pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")

result = pipeline.predict("file.pdf")
for res in result:
    res.print()
    res.save_to_json("output")
    res.save_to_img("output")