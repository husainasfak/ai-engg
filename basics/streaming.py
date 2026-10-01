
from openai import OpenAI, APIError
import os 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(api_key=openai_api_key)


stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain how a hash table works in 3 sentences."},
    ],
    stream=True,
    max_completion_tokens=200,
)



full_response = ""

for chunk in stream:
    if not chunk.choices:
        continue
    
    delta = chunk.choices[0].delta
    content = delta.content
    if content:
        print(content, end="", flush=True)
        full_response += content
    
    finish_reason = chunk.choices[0].finish_reason
    if finish_reason:
        print(f"\n\nFinish reason: {finish_reason}")
        
        
# stream=True returns an iterator of chunks instead of one completed response.
# Streaming chunks use delta, not message, because each chunk contains only the new piece.
# Some chunks carry metadata instead of text, so delta.content can be empty.
# flush=True tells Python to print immediately instead of buffering output.
# finish_reason tells you why generation stopped. "stop" means normal completion. "length" means the output limit was reached.

print(f"Full response length: {len(full_response)} characters")

# Handling Different Chunk Types
# Not every chunk contains displayable text. Some chunks contain role information, finish information, usage data, or provider metadata.
for chunk in stream:
    choice = chunk.choices[0] if chunk.choices else None

    if choice and choice.delta.role:
        print(f"[Role]: {choice.delta.role}")

    if choice and choice.delta.content:
        print(f"[Content]: {choice.delta.content}")

    if choice and choice.finish_reason:
        print(f"[Done]: finish_reason={choice.finish_reason}")

    usage = getattr(chunk, "usage", None)
    if usage:
        print(
            f"[Usage]: {usage.prompt_tokens} prompt + "
            f"{usage.completion_tokens} completion = "
            f"{usage.total_tokens} total tokens"
        )
        
# stream_options={"include_usage": True} asks for token usage in the final streaming chunk. Provider support can vary, so production systems should still handle missing usage data.

#  Stream Cancellation - when user click stop or we need to cancel the stream in any situtation


collected = ""
try:
    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = delta.content
        if content:
            collected += content
            print(content, end="", flush=True)

        if len(collected) > 200:
            print("\n\n[Cancelled: got enough text]")
            break
finally:
    stream.close()

print(f"Collected {len(collected)} characters before cancelling.")
# The finally block makes sure the connection is closed even if you break early or hit an exception.



# ERROR HANDLEING

# Pre stream errors
try:
    stream = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        max_completion_tokens=100,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
except APIError as error:
    print(f"API error before streaming started: {error.status_code} - {error.message}")
    
    

# Mid-Stream Errors
try:
    stream = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": "Explain recursion in detail."}],
        stream=True,
        max_completion_tokens=200,
    )

    for chunk in stream:
        stream_error = getattr(chunk, "error", None)
        if stream_error:
            message = getattr(stream_error, "message", stream_error)
            print(f"\n[Stream error]: {message}")
            error_occurred = True
            break

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if choice.finish_reason == "error":
            print("\n[Stream error]: generation stopped with finish_reason='error'")
            error_occurred = True
            break

        content = choice.delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content

except APIError as error:
    print(f"API error: {error.status_code} - {error.message}")
    error_occurred = True

if error_occurred and full_response:
    print(f"\n[Partial response collected: {len(full_response)} characters]")
    
    
    
