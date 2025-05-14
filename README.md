# 说明
FastAPI集成PaddleOCR识别功能  
支持传入图片以及pdf文件  
最低 Python 环境版本推荐为 3.8

# 安装方法
1. 在根目录创建虚拟环境
```bash
python -m venv PaddleOCRFastAPI
```
2. 激活虚拟环境  
- Windows（cmd）
```cmd
PaddleOCRFastAPI\Scripts\activate
```
- Windows（PowerShell）
```PowerShell
PaddleOCRFastAPI\Scripts\Activate.ps1
```
- macOS / Linux（bash/zsh）
```bash
source PaddleOCRFastAPI/bin/activate
```
- 退出虚拟环境（所有平台通用）
```bash
deactivate
```
3. 安装依赖工具
- 打开官网根据需要使用的版本进行安装必须依赖
[https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)
- 安装完成后手动其他依赖
```bash
pip install fastapi uvicorn python-dotenv numpy opencv-python PyMuPDF Pillow
```

4. 下载模型文件
启动项目后会自动下载模型到 `app/model`文件夹的指定位置  
也可以在 [PP-OCR系列模型列表](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/model_list.html) 中手动下载模型并放入 `app/model`文件夹的指定位置  
 .env 提供了`全量模型` 和 `超轻量` 两种模型，默认使用 `全量模型` 可自主注释内容进行选择  
如若使用其他模型需将 .env 文件中的 `# DET_MODEL_DIR` `REC_MODEL_DIR` `CLS_MODEL_DIR` 修改为实际使用的模型名称  

5. 启动项目  
项目服务端口为： 127.0.0.1:8888  
默认仅限本机进行访问，如果有需要可以修改启动文件或者使用命令行启动
- 双击启动
```
windows 双击 start.bat
macOS   双击 start.command
```
- 命令行启动  
```bash
uvicorn app.start:app --reload --host 127.0.0.1 --port 8888
```

# 使用方法
以下是提供的两个接口信息

| 接口信息          | /ocr_pdf (提取PDF文字)                          | /ocr_image (提取图片文字)                     |
|-------------------|------------------------------------------------|---------------------------------------------|
| **请求方法**      | POST                                           | POST                                        |
| **请求参数**      | file: UploadFile (PDF文件)                     | file: UploadFile (图片文件)                 |
| **响应类型**      | PlainTextResponse (纯文本)                     | PlainTextResponse (纯文本)                  |
| **功能描述**      | 提取PDF文档中的所有文字（按页输出）             | 提取图片中的文字                            |
| **文件类型限制**  | .pdf 文件                                      | 常见图片格式(.jpg, .png, .bmp等)            |
| **处理流程**      | 1. PDF转图片<br>2. 逐页OCR识别<br>3. 合并结果  | 1. 直接对图片进行OCR识别                    |
| **返回格式**      | 按页分组的识别文字（带页码标识）                | 直接输出图片中的所有识别文字                |
| **典型响应示例**  | `第1页：\n文本行1\n文本行2\n\n第2页：\n文本行3` | `识别文字1\n识别文字2\n识别文字3`           |
| **适用场景**      | 多页文档文字提取                               | 单张图片文字提取                           |
