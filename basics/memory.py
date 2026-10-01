import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(api_key=openai_api_key)

def chat():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful programming assistant. Give concise, practical answers.",
        }
    ]

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_completion_tokens=300,
        )

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"\nAssistant: {assistant_message}\n")

    print(f"\nConversation ended. Total messages: {len(messages)}")

chat()


# Context Window Limits
# A context window is the maximum number of tokens a model can process in one request. It includes the system message, conversation history, retrieved documents, tool results, and the model's response.


# Truncation Strategies -  when history gets too large

# Strategy 1: Drop Oldest Messages - Always preserve the system message. This is fast and cheap, but it can break continuity. If the user refers to something from an early turn, the model may no longer have that information.
def truncate_oldest(messages: list[dict], max_tokens: int, count_fn) -> list[dict]:
    """Remove old messages until the request fits."""
    system_msg = [messages[0]] if messages[0]["role"] == "system" else []
    conversation = messages[1:] if system_msg else messages[:]

    while count_fn(system_msg + conversation) > max_tokens and len(conversation) > 1:
        conversation.pop(0)

    return system_msg + conversation


# Strategy 2: Sliding Window with Pair Removal
# With pair removal, you trim the history in matched pairs:

# remove the oldest user message
# also remove the assistant reply to that message
# repeat until the conversation fits

def truncate_pairs(messages: list[dict], max_tokens: int, count_fn) -> list[dict]:
    """Remove oldest user/assistant pairs until under the token limit."""
    system_msg = []
    conversation = messages[:]

    if conversation and conversation[0]["role"] == "system":
        system_msg = [conversation.pop(0)]

    while count_fn(system_msg + conversation) > max_tokens and len(conversation) > 2:
        conversation.pop(0)
        if conversation and conversation[0]["role"] == "assistant":
            conversation.pop(0)

    return system_msg + conversation


# Strategy 3: Summarization
def summarize_old_messages(messages: list[dict], keep_recent: int = 6) -> list[dict]:
    """Summarize older messages and keep recent ones intact."""
    system_msg = []
    conversation = messages[:]

    if conversation and conversation[0]["role"] == "system":
        system_msg = [conversation.pop(0)]

    if len(conversation) <= keep_recent:
        return messages

    old_messages = conversation[:-keep_recent]
    recent_messages = conversation[-keep_recent:]

    summary_prompt = (
        "Summarize the following conversation for future context. "
        "Focus on key facts, decisions, and context that might be "
        "referenced later:\n\n"
    )
    for msg in old_messages:
        summary_prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

    summary_response = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_completion_tokens=300,
    )

    summary_text = summary_response.choices[0].message.content

    summary_message = {
        "role": "system",
        "content": f"Summary of earlier conversation: {summary_text}",
    }

    return system_msg + [summary_message] + recent_messages



# Token-Aware Conversation Windowing
