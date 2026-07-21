"""Constants for the Chinese companion service."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Qwen-compatible endpoint. Keep machine-specific addresses in .env.
QWEN_URL = os.getenv("QWEN_URL", "http://127.0.0.1:11434/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5:4b")
# The 4B deployment defaults to visible reasoning.  Companion replies should
# be concise and conversational, so disable it through vLLM's chat template.
QWEN_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Set to True to use DeepSeek for generation (much better Chinese, but uses API credits)
USE_DEEPSEEK_GEN = os.getenv("USE_DEEPSEEK_GEN", "").lower() in ("1", "true", "yes")

# Backend used only for extracting user-confirmed memories. Default to the
# already configured Qwen service when no paid DeepSeek key is selected.
MEMORY_EXTRACTION_BACKEND = os.getenv("MEMORY_EXTRACTION_BACKEND", "qwen").lower()

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chat_logs.db")
# User-controlled memories and reminders must be writable by this service.
# Keep them separate from legacy shared chat logs, whose directory may be
# mounted read-only in deployments.
STATE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "companion_state.db")

# A compact history keeps the 4B model focused and improves first-token latency.
MAX_CONTEXT = 4
PORT = 8016
DEFAULT_CITY = "北京"              # default city for weather queries

# Completed chat record reporting. The browser only signals that a session has
# ended; the backend reads the authoritative rounds from SQLite and uploads once.
CHAT_RECORD_ENABLED = os.getenv("CHAT_RECORD_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
)
CHAT_RECORD_ENDPOINT = os.getenv(
    "CHAT_RECORD_ENDPOINT",
    "",
)
CHAT_RECORD_RESIDENT_NAME = os.getenv("CHAT_RECORD_RESIDENT_NAME", "")
CHAT_RECORD_ROOM_NO = os.getenv("CHAT_RECORD_ROOM_NO", "")
CHAT_RECORD_TIMEOUT_SECONDS = float(os.getenv("CHAT_RECORD_TIMEOUT_SECONDS", "8"))
CHAT_RECORD_MAX_RETRIES = int(os.getenv("CHAT_RECORD_MAX_RETRIES", "2"))
CHAT_RECORD_API_TOKEN = os.getenv("CHAT_RECORD_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Dialect / colloquial styles — selectable via API & frontend
# ---------------------------------------------------------------------------
DIALECT_STYLES = {
    "northern": {
        "label": "北方口语",
        "flavor": (
            "说话带点北方口语的亲切劲儿。像「可真是」「那可不」「可不是嘛」「得嘞」这些词，"
            "挑着用，换着来，别每句都同一个口头禅。自然就好，别刻意。\n"
        ),
    },
    "dongbei": {
        "label": "东北口语",
        "flavor": (
            "说话带点东北口语的热乎劲儿。像「可拉倒吧」「那可不咋的」「老好了」「嗯呐」这些词，"
            "挑着用，换着来，别每句都同一个口头禅。自然就好，别刻意。\n"
        ),
    },
    "sichuan": {
        "label": "四川口语",
        "flavor": (
            "说话带点四川口语的味道。像「要得」「巴适」「咋子嘛」「硬是」「对头」这些词，"
            "挑着用，换着来，别每句都同一个口头禅。自然就好，别刻意。\n"
        ),
    },
    "jiangzhe": {
        "label": "江浙口语",
        "flavor": (
            "说话带点江浙口语的软和劲儿。像「好的呀」「是哦」「蛮好」「对哇」「好的啦」这些词，"
            "挑着用，换着来，别每句都同一个口头禅。自然就好，别刻意。\n"
        ),
    },
    "standard": {
        "label": "通用口语",
        "flavor": (
            "说自然的普通话口语，不刻意带地方味儿。像日常聊天一样，"
            "用「挺好的」「是吧」「对呀」「嗯嗯」这种自然的词。别每句都同一个口头禅。\n"
        ),
    },
}
