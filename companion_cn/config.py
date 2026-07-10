"""Constants for the Chinese companion service."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

QWEN_URL = "http://192.168.253.91:8003/v1"
QWEN_MODEL = "/model"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Set to True to use DeepSeek for generation (much better Chinese, but uses API credits)
USE_DEEPSEEK_GEN = os.getenv("USE_DEEPSEEK_GEN", "").lower() in ("1", "true", "yes")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "arrowcanaria_server", "chat_logs.db")

MAX_CONTEXT = 6
PORT = 8016
DEFAULT_CITY = "北京"              # default city for weather queries

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
