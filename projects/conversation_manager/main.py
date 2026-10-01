import os
import tiktoken
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()




class ConversationManager:
    """Manage conversation history with token-aware truncation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_tokens: int = 400_000,
        max_response_tokens: int = 4_096,
        system_prompt: str = "You are a helpful assistant.",
        truncation_strategy: str = "pairs",
    ):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        self.truncation_strategy = truncation_strategy
        self.client = OpenAI(api_key=openai_api_key)
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.messages = [{"role": "system", "content": system_prompt}]

    def count_tokens(self, messages: list[dict] = None) -> int:
        """Estimate tokens in the given messages or current history."""
        msgs = messages or self.messages
        tokens = 0
        for message in msgs:
            tokens += 4
            for key, value in message.items():
                tokens += len(self.encoding.encode(value))
        tokens += 2
        return tokens

    @property
    def available_tokens(self) -> int:
        """Tokens available for input after reserving room for output."""
        return self.max_context_tokens - self.max_response_tokens

    def _truncate_if_needed(self):
        if self.count_tokens() <= self.available_tokens:
            return

        if self.truncation_strategy == "oldest":
            self._truncate_oldest()
        elif self.truncation_strategy == "pairs":
            self._truncate_pairs()
        elif self.truncation_strategy == "summarize":
            self._summarize_old()

    def _truncate_oldest(self):
        """Drop oldest non-system messages one at a time."""
        while self.count_tokens() > self.available_tokens and len(self.messages) > 2:
            self.messages.pop(1)

    def _truncate_pairs(self):
        """Drop oldest user/assistant pairs."""
        while self.count_tokens() > self.available_tokens and len(self.messages) > 3:
            removed = self.messages.pop(1)
            if (removed["role"] == "user"
                    and len(self.messages) > 1
                    and self.messages[1]["role"] == "assistant"):
                self.messages.pop(1)

    def _summarize_old(self):
        """Summarize older messages, keep recent ones."""
        keep_recent = 6

        if len(self.messages) <= keep_recent + 1:
            self._truncate_pairs()
            return

        system_msg = self.messages[0]
        old_msgs = self.messages[1:-keep_recent]
        recent_msgs = self.messages[-keep_recent:]

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_msgs
        )

        summary_response = self.client.chat.completions.create(
            model="openai/gpt-5.4-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation for future context. "
                    f"Focus on key facts and decisions:\n\n{conversation_text}"
                ),
            }],
            max_completion_tokens=200,
        )

        summary = summary_response.choices[0].message.content

        self.messages = [
            system_msg,
            {"role": "system", "content": f"Earlier conversation summary: {summary}"},
            *recent_msgs,
        ]

    def send(self, user_message: str, stream: bool = True) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self._truncate_if_needed()

        if stream:
            return self._send_streaming()
        else:
            return self._send_normal()

    def _send_streaming(self) -> str:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_completion_tokens=self.max_response_tokens,
            stream=True,
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

        print()
        self.messages.append({"role": "assistant", "content": full_response})
        return full_response

    def _send_normal(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_completion_tokens=self.max_response_tokens,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def get_stats(self) -> dict:
        token_count = self.count_tokens()
        return {
            "message_count": len(self.messages),
            "token_count": token_count,
            "available_tokens": self.available_tokens,
            "usage_percent": round(token_count / self.available_tokens * 100, 1),
        }
        

manager = ConversationManager(
    model="gpt-4o-mini",
    max_context_tokens=8_000,  # Small on purpose so truncation is easy to test
    max_response_tokens=1_000,
    system_prompt="You are a Python tutor. Be concise.",
    truncation_strategy="pairs",
)

topics = [
    "What is a list comprehension?",
    "Show me a nested list comprehension example.",
    "How do generator expressions differ from list comprehensions?",
    "What about dictionary comprehensions?",
    "When should I avoid comprehensions?",
    "Can you remind me what we discussed about generators?",
]

for topic in topics:
    print(f"\nYou: {topic}")
    print(f"Assistant: ", end="")
    manager.send(topic, stream=True)

    stats = manager.get_stats()
    print(f"  [{stats['message_count']} messages, "
          f"{stats['token_count']} tokens, "
          f"{stats['usage_percent']}% used]")