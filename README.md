# 说明

本项目基于 FastAPI 集成 PaddleOCR，实现图片和 PDF 文件的文字识别。  
推荐最低 Python 版本为 3.8。

# 安装步骤


1. 安装依赖
* 先访问官网安装 PaddlePaddle（根据系统和环境选择合适版本）
  [https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)
  
* 安装其他依赖：
    ```bash
    pip install -r requirements.txt
    ```

2. 模型文件下载
项目仓库中已提供以全量模型和轻量模型，无需下载。

3. 通过命令行启动项目
    ```bash
    uvicorn start:app --host 127.0.0.1 --port 8888 --reload
    ```
* 如需允许局域网访问，可修改启动命令，将 `127.0.0.1` 改为 `0.0.0.0`

---

# 接口文档

#### 基本信息
- **URL**: `/ocr`
- **HTTP 方法**: `POST`
- **Content-Type**: `multipart/form-data`
- **响应格式**: `application/json`

#### 请求参数

| 参数名      | 类型       | 是否必须 | 说明                     |
|-------------|------------|----------|--------------------------|
| file        | 文件(File) | 是       | 需要上传的图片或PDF文件。支持的图片类型包括JPEG、PNG等；支持PDF文件。 |
| result_type | 字符串(Str) | 是       | 结果返回类型。<br>可选值：`return_final_result_one_line`, `return_final_result`, `return_result`, `return_result_with_result_photo`。分别对应不分行显示结果、分行显示结果、分行显示数据、分行显示数据并附带标注图片。 |
| model       | 字符串(Str) | 是       | 使用的OCR模型。<br>可选值：`ocr_light`（轻量模型），`ocr_server`（全量模型）。 |

#### 返回字段

- **result**: 根据所选的`result_type`不同，返回的结果格式会有所不同。
    - 如果选择了`return_final_result_one_line`，则返回一个字符串，代表所有文字连接成的一行文本。
    - 如果选择了`return_final_result`，则返回一个数组，每个元素为一行文本。
    - 如果选择了`return_result`，则返回一个数组，每个元素是一个包含位置和文本的元组。
    - 如果选择了`return_result_with_result_photo`，除了上述结果外，还会额外返回标注图片的base64编码列表。
- **image_base64_list**: 当且仅当`result_type`为`return_result_with_result_photo`时出现，包含每页处理后的图像的base64编码。
- **detail_time**: 处理所需时间（秒）。
- **error**: 发生错误时返回的信息（如果有）。

#### 示例请求

```bash
curl -X POST "http://127.0.0.1:8888/ocr" \
-H "accept: application/json" \
-F "file=@/path/to/your/file.png" \
-F "result_type=return_result_with_result_photo" \
-F "model=ocr_light"
```

#### 示例响应

成功情况下（以`return_result_with_result_photo`为例）：

```json
{
  "result": [
    [[430.0, 59.0], [559.0, 59.0], [559.0, 93.0], [430.0, 93.0]],
    "识别结果示例",
    0.998164
  ],
  "image_base64_list": ["base64_encoded_image_string_here"],
  "detail_time": 1.234
}
```

错误情况下：

```json
{
  "error": "无效的 result_type 参数",
  "detail_time": 0.002
}
```