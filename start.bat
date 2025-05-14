@echo off
uvicorn app.start:app --reload --host 127.0.0.1 --port 8888
pause
