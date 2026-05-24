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

temp_env = os.getenv("LLM_TEMPERATURE")
TEMPERATURE = float(temp_env) if temp_env and temp_env.strip() else 0.1

max_tok_env = os.getenv("LLM_MAX_TOKENS")
MAX_TOKENS = int(max_tok_env) if max_tok_env and max_tok_env.strip() else 8192

# Backward compatibility for older code that used ENDPOINT as the model name.
ENDPOINT = MODEL
