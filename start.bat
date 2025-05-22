@echo off
cd /d %~dp0
uvicorn ocr_server:app --host 127.0.0.1 --port 8000 --reload
pause
