"""
Build Your Own Agent Harness - V1: A Single Message

The smallest possible harness: send ONE message to the model and print the
reply. No loop, no memory, no tools. Every later version is built around
this one API call.
"""

from config import MODEL, client

MAX_TOKENS = 16000


def main():
    user_input = input("you > ")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": user_input}],
    )

    # The reply is a list of content blocks. For now we only need the text.
    for block in response.content:
        if block.type == "text":
            print(f"\nclaude > {block.text}\n")

    print(f"[{response.usage.input_tokens} tokens in, {response.usage.output_tokens} tokens out]")


if __name__ == "__main__":
    main()
