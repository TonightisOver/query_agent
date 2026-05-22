import os
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("MOONSHOT_API_KEY")
    or os.getenv("DOUBAO_API_KEY")
)

BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("MOONSHOT_BASE_URL")
    or os.getenv("DOUBAO_BASE_URL")
    or "https://ark.cn-beijing.volces.com/api/v3"
)

MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("MOONSHOT_MODEL")
    or os.getenv("DOUBAO_MODEL")
    or os.getenv("DOUBAO_ENDPOINT")
)

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))

# Backward compatibility for older code that used ENDPOINT as the model name.
ENDPOINT = MODEL
