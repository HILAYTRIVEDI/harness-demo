"""
Build Your Own Agent Harness - V2: The Chat Loop

V1 sent one message and forgot it. V2 wraps that call in a loop:
  1. read what the user types
  2. append it to the conversation history
  3. send the WHOLE history to the model
  4. append the model's reply to the history
  5. repeat

The model is stateless. The harness is what gives it memory.
"""

from config import MODEL, client

MAX_TOKENS = 16000


def main():
    messages = []  # the entire conversation lives here
    print("V2 harness - type 'exit' to quit\n")

    while True:
        user_input = input("you > ").strip()
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )

        # Keep the full content (not just the text) so the history stays exact.
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nclaude > {block.text}\n")


if __name__ == "__main__":
    main()
