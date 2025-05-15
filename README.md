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
pip install fastapi uvicorn paddleocr opencv-python numpy pydantic python-multipart pymupdf pillow
```

4. 下载模型文件  
项目启动默认下载`全量模型` 和 `轻量模型` 两种模型，不同接口调用不同的模型  
如若使用其他模型修改 `DET_MODEL_DIR` `REC_MODEL_DIR` `CLS_MODEL_DIR` 等变量即可 

5. 启动命令
```
python3 start.py
```
项目服务端口为： 127.0.0.1:8888  
默认仅限本机进行访问，如果有需要可以修改 `start.py`文件中最后一行的启动命令

# 接口信息
## 图片接口

| 请求路径 | /ocr\_image\_light         | /ocr\_image\_full         |
| ---- | -------------------------- | ------------------------- |
| 接口名称 | 提取图片文字——超轻量                | 提取图片文字——全量模型              |
| 请求方式 | POST                       | POST                      |
| 返回内容 | 识别出的文字文本                   | 识别出的文字文本                  |
| 请求参数 | file: 图片文件 (UploadFile)    | file: 图片文件 (UploadFile)   |
| 备注   | 使用轻量级OCR模型，带文字检测（det=True） | 使用全量OCR模型，带文字检测（det=True） |

## pdf接口
| 请求路径 | /ocr\_pdf\_light           | /ocr\_pdf\_full           |
| ---- |----------------------------|---------------------------|
| 接口名称 | 提取pdf文字——超轻量               | 提取pdf文字——全量模型             |
| 请求方式 | POST                       | POST                      |
| 返回内容 | 识别出的文字文本                   | 识别出的文字文本                  |
| 请求参数 | file: pdf文件 (UploadFile)   | file: pdf文件 (UploadFile)  |
| 备注   | 使用轻量级OCR模型，带文字检测（det=True） | 使用全量OCR模型，带文字检测（det=True） |
