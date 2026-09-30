import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(api_key=openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a senior software engineer. Give concise, practical answers.",
        },
        {
            "role": "user",
            "content": "What are the three most important things to know about database indexing?",
        },
    ],
    max_completion_tokens=300,
)

answer = response.choices[0].message.content
print(answer)

if response.usage:
    print(
        f"\nTokens used - Input: {response.usage.prompt_tokens}, "
        f"Output: {response.usage.completion_tokens}, "
        f"Total: {response.usage.total_tokens}"
    )