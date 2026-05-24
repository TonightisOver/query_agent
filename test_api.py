import os
from openai import OpenAI
from config import API_KEY, ENDPOINT, MODEL, BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def chat(prompt: str):
    response = client.chat.completions.create(
        model=ENDPOINT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    result = chat("你好")
    print(result)
