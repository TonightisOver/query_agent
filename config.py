import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DOUBAO_API_KEY")
ENDPOINT = os.getenv("DOUBAO_ENDPOINT")
MODEL = os.getenv("DOUBAO_MODEL")

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
