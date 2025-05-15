@echo off
cd /d %~dp0
uvicorn start:app --host 127.0.0.1 --port 8888 --reload=true
pause
