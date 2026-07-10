"""Chinese companion — FastAPI entry point (:8016)."""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from .api import router
from .openai_compat import router as openai_router
from .config import PORT

app = FastAPI(title="养老陪伴", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Internal API (legacy)
app.include_router(router)
# OpenAI-compatible API (for external clients)
app.include_router(openai_router)

# Serve frontend (must be last to avoid route conflicts)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print(f"养老陪伴服务启动 → http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
