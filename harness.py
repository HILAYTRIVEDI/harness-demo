"""
Build Your Own Agent Harness - V3: Tools + The Agent Loop

V2 could only talk. V3 can act.

The agent loop:
  1. send history + tool schemas to the model
  2. if the model answers with tool_use blocks, the harness runs them
  3. send the tool results back as a user message
  4. repeat until the model stops asking for tools (stop_reason == "end_turn")

One user message can now trigger many model calls.
"""

import json

from config import MODEL, client
from tools import TOOLS, run_tool

MAX_TOKENS = 16000


def agent_turn(messages):
    """Run the agent loop until the model is done with this user request."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nclaude > {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        # The model asked for one or more tools. Run them all, then send
        # every result back together in ONE user message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({json.dumps(block.input)})")
            output = run_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # links the result to the request
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})


def main():
    messages = []
    print("V3 harness - type 'exit' to quit\n")

    while True:
        user_input = input("you > ").strip()
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        agent_turn(messages)


if __name__ == "__main__":
    main()
