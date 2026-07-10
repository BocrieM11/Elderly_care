"""Constants — one place for everything that might change."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

ARROWCANARIA_URL = "http://127.0.0.1:8014/v1"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "arrowcanaria_server", "chat_logs.db")

MAX_CONTEXT = 6          # conversation rounds to keep
PORT = 8015
