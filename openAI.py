import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
)

output = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Input text",
        }
    ],
    model="Codex",
)

print(output)