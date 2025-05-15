# 说明

本项目基于 FastAPI 集成 PaddleOCR，实现图片和 PDF 文件的文字识别。  
推荐最低 Python 版本为 3.8。

# 安装步骤

1. **创建虚拟环境**（在项目根目录执行）：

```bash
python -m venv PaddleOCRFastAPI
```

2. **激活虚拟环境**

* Windows (cmd)：

```cmd
PaddleOCRFastAPI\Scripts\activate
```

* Windows (PowerShell)：

```powershell
PaddleOCRFastAPI\Scripts\Activate.ps1
```

* macOS / Linux (bash/zsh)：

```bash
source PaddleOCRFastAPI/bin/activate
```

* **退出虚拟环境（所有平台通用）**：

```bash
deactivate
```

3. **安装依赖**

* 先访问官网安装 PaddlePaddle（根据系统和环境选择合适版本）
  [https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)
* 安装其他依赖：

```bash
pip install fastapi uvicorn paddleocr opencv-python numpy pydantic python-multipart pymupdf pillow
```

4. **模型文件下载**  
   项目默认自动下载两种模型：

* **全量模型**
* **轻量模型**
  不同接口调用不同模型。  
  如果需要使用自定义模型，请修改代码中 `DET_MODEL_DIR`、`REC_MODEL_DIR`、`CLS_MODEL_DIR` 等变量。

5. **启动项目**

* 可通过双击平台对应的启动脚本启动
* 或通过命令行启动（开发时建议加 `--reload` 热重载）：

```bash
uvicorn start:app --host 127.0.0.1 --port 8888 --reload=true
```

* 默认服务监听地址和端口为 `127.0.0.1:8888`，只允许本机访问。
* 如需允许局域网访问，可修改启动命令或代码，将 `host` 改为 `0.0.0.0`。

---

# 接口信息

## 图片识别接口

| 请求路径 | /ocr\_image\_light         | /ocr\_image\_full         |
| ---- | -------------------------- | ------------------------- |
| 接口名称 | 图片文字提取——超轻量模型              | 图片文字提取——全量模型              |
| 请求方式 | POST                       | POST                      |
| 请求参数 | file: 图片文件（UploadFile）     | file: 图片文件（UploadFile）    |
| 返回内容 | 识别出的文字文本                   | 识别出的文字文本                  |
| 备注   | 使用轻量级OCR模型，带文字检测（det=True） | 使用全量OCR模型，带文字检测（det=True） |

## PDF识别接口

| 请求路径 | /ocr\_pdf\_light           | /ocr\_pdf\_full           |
| ---- | -------------------------- | ------------------------- |
| 接口名称 | PDF文字提取——超轻量模型             | PDF文字提取——全量模型             |
| 请求方式 | POST                       | POST                      |
| 请求参数 | file: PDF文件（UploadFile）    | file: PDF文件（UploadFile）   |
| 返回内容 | 识别出的文字文本                   | 识别出的文字文本                  |
| 备注   | 使用轻量级OCR模型，带文字检测（det=True） | 使用全量OCR模型，带文字检测（det=True） |

