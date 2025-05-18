from pathlib import Path

from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(
    det_model_dir="../model/det/ch_PP-OCRv4_det_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)

ocr_full = PaddleOCR(
    det_model_dir="../model/det/ch_PP-OCRv4_det_server_infer",
    rec_model_dir="../model/rec/ch_PP-OCRv4_rec_server_infer",
    cls_model_dir="../model/cls/ch_ppocr_mobile_v2.0_cls_infer",
    lang="ch"
)


# 一行输出最终结果
def print_final_result_one_line(ocr_result):
    texts = [line[1][0] for line in ocr_result[0]]
    print(" ".join(texts).replace(" ", ""))


# 按行输出最终结果
def print_final_result(ocr_result):
    for line in ocr_result[0]:
        texts = line[1][0]
        print(texts)


# 输出坐标、结果、准确率
def print_result(ocr_result):
    for line in ocr_result[0]:
        print(line)


# 输出坐标、结果、准确率 + 标注图像的识别结果
def print_result_with_result_photo(ocr_result, image_path):
    # 使用函数 print_result 输出坐标、结果、准确率
    print_result(ocr_result)

    # 标注图像的识别结果
    ocr_result = ocr_result[0]
    image = Image.open(image_path).convert('RGB')
    boxes = [line[0] for line in ocr_result]
    txts = [line[1][0] for line in ocr_result]
    scores = [line[1][1] for line in ocr_result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='font/simfang.ttf')
    im_show = Image.fromarray(im_show)
    output_path = Path(image_path).stem + '_result.jpg'
    im_show.save(output_path)


# 检查需要 OCR 文件的存在
def check_ocr_file(file_path):
    file_path = Path(file_path)

    if not file_path.is_file():
        print(f"文件不存在: {file_path}")
        exit(1)


# 定义全局结果输出方式
def print_result_mode(ocr_result, image_path):
    # 检查输出内容
    if not ocr_result:
        print("未检测到文本")
        exit(1)

    # 一行输出最终结果
    print_final_result_one_line(ocr_result)
    # 按行输出最终结果
    print_final_result(ocr_result)
    # 按行输出坐标、结果、准确率
    print_result(ocr_result)
    # 按行输出坐标、结果、准确率 + 标注图像的识别结果
    print_result_with_result_photo(ocr_result, image_path)


# 定义全局 OCR 引擎，默认使用 中英文超轻量 PP-OCRv4 模型
# 使用 中英文超轻量 PP-OCRv4 模型
ocr_engine = ocr
# 使用 原始高精度模型 PP-OCRv4 模型
# ocr_engine = ocr_full


# 主函数
if __name__ == '__main__':
    try:
        # 定义要识别的文件的位置
        ocr_file_path = 'ocr_file_path.jpg'
        # 检查需要 OCR 文件的存在
        check_ocr_file(ocr_file_path)

        # 使用 paddleOCR 识别图片
        result = ocr_engine.ocr(ocr_file_path, det=True, cls=False)

        # 输出结果
        print_result_mode(result, ocr_file_path)
    except Exception as e:
        print(f"发生错误: {e}")
