#!/bin/bash
cd "$(dirname "$0")"
uvicorn start:app --host 127.0.0.1 --port 8888 --reload=true
