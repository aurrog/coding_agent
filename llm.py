from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv('API_KEY')
BASE_URL=os.getenv('BASE_URL')

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

def chat(messages):

    resp = client.chat.completions.create(
        model="deepseek/deepseek-v4-flash",        # или anthropic/claude-3.5-sonnet
        messages=[{"role": "user", "content": "Привет!"}],
    )
    return resp.choices[0].message.content
