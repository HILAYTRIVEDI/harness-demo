"""
Build Your Own Agent Harness - V5: Safety & Reliability

New in V5:
  - permission prompts before any tool that changes files or runs commands
  - a turn limit, so a confused agent can't loop forever
  - every stop_reason is handled (tool_use, end_turn, max_tokens, refusal)
  - API errors and Ctrl+C roll the conversation back instead of crashing
  - persistent memory: MEMORY.md is loaded into the system prompt at startup,
    and the `remember` tool appends to it (see memory.py)
"""

import json

import anthropic

from config import MODEL, client
from prompts import build_system_prompt
from tools import DANGEROUS_TOOLS, TOOLS, run_tool

MAX_TOKENS = 64000
MAX_TURNS = 25  # model calls allowed per user request

always_allowed = set()  # tools the user approved for the rest of the session


def call_model(system, messages):
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


def ask_permission(name, tool_input):
    """Human in the loop: the user approves risky tool calls."""
    if name not in DANGEROUS_TOOLS or name in always_allowed:
        return True
    preview = tool_input.get("command") or tool_input.get("path")
    answer = input(f"  Allow {name}: {preview}?  [y]es / [n]o / [a]lways > ").strip().lower()
    if answer == "a":
        always_allowed.add(name)
        return True
    return answer == "y"


def execute_tools(response):
    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        print(f"  [tool] {block.name}({json.dumps(block.input)[:200]})")

        if ask_permission(block.name, block.input):
            output, is_error = run_tool(block.name, block.input)
        else:
            output, is_error = "The user denied this tool call. Ask them how to proceed.", True

        if is_error:
            print(f"  [error] {output[:200]}")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
            "is_error": is_error,
        })
    return tool_results


def agent_turn(system, messages):
    for _ in range(MAX_TURNS):
        response = call_model(system, messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")

        if response.stop_reason == "max_tokens":
            if not any(b.type == "tool_use" for b in response.content):
                print("  [warning] reply was cut off at max_tokens")
                return
            # A tool call was cut off mid-input: don't run it, tell the model.
            messages.append({"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": "Not run: your output hit max_tokens and this tool input was cut off. "
                               "Retry with smaller pieces.",
                    "is_error": True,
                }
                for b in response.content if b.type == "tool_use"
            ]})
            continue

        if response.stop_reason != "tool_use":
            return  # end_turn: the agent is done

        messages.append({"role": "user", "content": execute_tools(response)})

    print(f"  [warning] stopped after {MAX_TURNS} turns")


def main():
    system = build_system_prompt()
    messages = []
    print("V5 harness - type 'exit' to quit, Ctrl+C to interrupt the agent\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue

        checkpoint = len(messages)  # where this request starts in the history
        messages.append({"role": "user", "content": user_input})
        try:
            agent_turn(system, messages)
        except KeyboardInterrupt:
            print("\n  [interrupted] request cancelled")
            del messages[checkpoint:]
        except (anthropic.APIError, RuntimeError) as e:
            print(f"\n  [error] {e}")
            del messages[checkpoint:]


if __name__ == "__main__":
    main()
