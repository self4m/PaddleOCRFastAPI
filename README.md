# 说明

本项目基于 FastAPI 集成 PaddleOCR，实现图片和 PDF 文件的文字识别。   
使用2025年5月20日发布的新识别模型PP-OCRv5，测试发现识别精准度以及速度仅有提升  

# 识别说明  
需要注意的是未经过训练的模型更适合用于识别格式统一、字体规范的文本，例如：
>标准印刷体文字:   
> - 比如书籍、打印的文档、PDF 导出内容、清晰扫描件。 
> - 字体规范、对齐规则、间距正常，没有扭曲或遮挡。  
> 
> 无复杂排版格式:   
> - 纯文字或简单结构：如左对齐、段落整齐、不含多栏、表格、图片混排。
> 
> 背景干净、无噪声:   
> - 白底黑字最好，没有阴影、水印、涂改、褶皱等干扰。

如果需要识别提取特定格式的文本，例如票据、表格等，则需要进行数据标注然后训练专属的模型。

----

# 安装步骤


1. 安装依赖
* 先访问官网安装 PaddlePaddle（根据系统和环境选择合适版本）
  [https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)
  
* 安装其他依赖：
    ```bash
    pip install -r requirements.txt
    ```

2. 模型文件与配置  
项目仓库中已配置全量模型，无需下载。  
识别配置文件内容存储于 `ocr_config.yaml` 中  

3. 通过命令行启动项目
    ```bash
    uvicorn start:app --host 127.0.0.1 --port 8888 --reload
    ```
* 如需允许局域网访问，可修改启动命令，将 `127.0.0.1` 改为 `0.0.0.0`

---

# 接口文档

以下是基于你提供的代码生成的接口调用文档，该文档描述了如何调用 OCR 文字识别工具的 API 接口 `/ocr`。

### OCR 文字识别服务接口文档

#### 基本信息
- **URL**: `/ocr`
- **HTTP 方法**: `POST`
- **Content-Type**: `multipart/form-data`
- **响应格式**: `application/json`

#### 请求参数

| 参数名 | 类型       | 是否必须 | 说明                     |
|--------|------------|----------|--------------------------|
| file   | 文件(File) | 是       | 需要上传的图片或PDF文件。支持的图片类型包括JPEG、PNG等；支持PDF文件。 |

#### 返回字段

- **result**: 包含OCR识别的文字列表。
- **detail_time**: 处理所需时间（秒）。
- **image_list**: 每页处理后的图像的base64编码列表。每张图像是经过OCR标注后的结果图像。
- **error**: 发生错误时返回的信息（如果有）。

#### 示例请求

使用`curl`命令行工具进行测试：

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
-H "accept: application/json" \
-F "file=@/path/to/your/file.png"
```

请注意替换`/path/to/your/file.png`为实际的文件路径。

#### 示例响应

成功情况下：

```json
{
  "result": ["识别", "结果", "示例"],
  "detail_time": 1.234,
  "image_list": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAA...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAA..."
  ]
}
```

这里`image_list`中的每个元素是处理后图像的base64编码字符串，可以直接嵌入到HTML中显示。

错误情况下：

```json
{
  "error": "不支持的文件类型",
  "detail_time": 0.002
}
```

或者

```json
{
  "error": "文件处理失败: [具体错误信息]",
  "detail_time": 0.005
}
```

又或者

```json
{
  "error": "OCR识别失败: [具体错误信息]",
  "detail_time": 0.005
}
```
