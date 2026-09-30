import tiktoken
def main():
    def estimate_tokens(text: str) -> int:
        """Very rough estimate for English text."""
        return len(text) // 4
    
    prompt = "Explain the difference between TCP and UDP in networking."
    print(f"Estimated tokens: {estimate_tokens(prompt)}")
    
    prompt = "Explain the difference between TCP and UDP in networking."
    encoder = tiktoken.get_encoding("o200k_base")
    tokens = encoder.encode(prompt)
    print(f"Token count: {len(tokens)}")




if __name__ == "__main__":
    main()
