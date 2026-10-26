"""
Build Your Own Agent Harness - V4: System Prompt + Streaming

New in V4:
  - prompts.py builds a system prompt: the agent's role, rules and environment
  - responses stream token by token, so the user sees progress immediately
"""

import json

from config import MODEL, client
from prompts import build_system_prompt
from tools import TOOLS, run_tool

MAX_TOKENS = 64000  # streaming avoids HTTP timeouts, so we can allow long replies


def call_model(system, messages):
    """Stream one model response to the terminal and return the final message."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        printed_header = False
        for text in stream.text_stream:
            if not printed_header:
                print("\nclaude > ", end="", flush=True)
                printed_header = True
            print(text, end="", flush=True)
        if printed_header:
            print("\n")
        return stream.get_final_message()


def agent_turn(system, messages):
    while True:
        response = call_model(system, messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({json.dumps(block.input)})")
            output = run_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})


def main():
    system = build_system_prompt()  # built once, stays the same all session
    messages = []
    print("V4 harness - type 'exit' to quit\n")

    while True:
        user_input = input("you > ").strip()
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        agent_turn(system, messages)


if __name__ == "__main__":
    main()
