from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")