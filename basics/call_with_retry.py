import os
import random
import time
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError
)


load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(
    api_key=openai_api_key,
    timeout=30.0,
)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def retry_delay(attempt: int) -> float:
    return min((2 ** attempt) + random.uniform(0, 1), 10)


def call_with_retry(max_retries=3, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError:
            if attempt == max_retries:
                raise
            wait_time = retry_delay(attempt)
            print(f"Rate limited. Retrying in {wait_time:.1f}s.")
            time.sleep(wait_time)

        except (APITimeoutError, APIConnectionError):
            if attempt == max_retries:
                raise
            wait_time = retry_delay(attempt)
            print(f"Temporary connection problem. Retrying in {wait_time:.1f}s.")
            time.sleep(wait_time)

        except APIStatusError as error:
            if error.status_code not in RETRYABLE_STATUS_CODES or attempt == max_retries:
                raise
            wait_time = retry_delay(attempt)
            print(f"API error {error.status_code}. Retrying in {wait_time:.1f}s.")
            time.sleep(wait_time)



response = call_with_retry(
    max_retries=3,
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain TCP vs UDP in one paragraph."},
    ],
    max_completion_tokens=300,
)
print(response.choices[0].message.content)