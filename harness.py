"""
Build Your Own Agent Harness - V6: Context Management

The context window is the agent's working memory, and every turn resends
the whole history. New in V6:
  - token tracking: see how big the context is and what each call costs
  - auto-compaction: when the context grows too big, the model summarizes the
    conversation and the harness starts fresh from that summary
  - slash commands: type / to list them all, Tab to complete
    (/help /tokens /compact /clear /save /exit)
  - /save writes the transcript to sessions/ as JSON
"""

import datetime
import json
from pathlib import Path

import anthropic

from config import MODEL, client
from prompts import COMPACT_PROMPT, build_system_prompt
from tools import DANGEROUS_TOOLS, TOOLS, run_tool

MAX_TOKENS = 64000
MAX_TURNS = 25
COMPACT_AT = 150000  # compact once the context passes this many tokens

always_allowed = set()
stats = {"context": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0}


def call_model(system, messages, echo=True):
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        printed_header = False
        for text in stream.text_stream:
            if not echo:
                continue
            if not printed_header:
                print("\nclaude > ", end="", flush=True)
                printed_header = True
            print(text, end="", flush=True)
        if printed_header:
            print("\n")
        response = stream.get_final_message()

    track_usage(response.usage)
    return response


def track_usage(usage):
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    stats["input"] += usage.input_tokens
    stats["cache_read"] += cache_read
    stats["cache_write"] += cache_write
    stats["output"] += usage.output_tokens
    # Everything the model just read = the current size of the context.
    stats["context"] = usage.input_tokens + cache_read + cache_write


def print_stats():
    print(f"  context: {stats['context']:,} tokens (compacts at {COMPACT_AT:,})")
    print(f"  session: {stats['input']:,} input, {stats['cache_read']:,} cache read, "
          f"{stats['cache_write']:,} cache write, {stats['output']:,} output")


# ---------------------------------------------------------------------------
# Compaction: replace a long history with a short summary
# ---------------------------------------------------------------------------

def compact(system, messages):
    if not messages:
        return
    print("  [compacting conversation...]")
    request = messages + [{"role": "user", "content": COMPACT_PROMPT}]
    response = call_model(system, request, echo=False)
    summary = "".join(b.text for b in response.content if b.type == "text")

    messages[:] = [
        {"role": "user", "content": f"Summary of our conversation so far:\n\n{summary}"},
        {"role": "assistant", "content": "Got it. I'll continue from this summary."},
    ]
    print(f"  [compacted: {stats['context']:,} tokens -> summary of {len(summary):,} characters]")
    stats["context"] = len(summary) // 4  # rough estimate until the next model call


# ---------------------------------------------------------------------------
# Tools + the agent loop (same as V5)
# ---------------------------------------------------------------------------

def ask_permission(name, tool_input):
    if name not in DANGEROUS_TOOLS or name in always_allowed:
        return True
    preview = tool_input.get("command") or tool_input.get("path") or tool_input.get("fact")
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
            return

        messages.append({"role": "user", "content": execute_tools(response)})

    print(f"  [warning] stopped after {MAX_TURNS} turns")


# ---------------------------------------------------------------------------
# Sessions + slash commands
# ---------------------------------------------------------------------------

def to_json(content):
    """SDK content blocks are objects; turn them into plain dicts for JSON."""
    if isinstance(content, str):
        return content
    return [b if isinstance(b, dict) else b.model_dump(exclude_none=True) for b in content]


def save_session(messages):
    Path("sessions").mkdir(exist_ok=True)
    path = Path("sessions") / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    data = [{"role": m["role"], "content": to_json(m["content"])} for m in messages]
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved {len(messages)} messages to {path}")


# One list drives the command menu, the matching and Tab completion,
# so adding a command means adding one line here (and its branch below).
COMMANDS = {
    "/help": "list all commands",
    "/tokens": "show context size and token usage",
    "/compact": "summarize the conversation to free up context",
    "/clear": "start a fresh conversation",
    "/save": "save the transcript to sessions/",
    "/exit": "quit",
}


def list_commands(prefix="/"):
    """Print every command that starts with prefix. Returns how many matched."""
    matches = [name for name in COMMANDS if name.startswith(prefix)]
    for name in matches:
        print(f"  {name:<10}{COMMANDS[name]}")
    return len(matches)


def enable_tab_completion():
    """Tab-complete slash commands, like the command menu in Claude Code."""
    try:
        import readline
    except ImportError:  # no readline (e.g. plain Windows): typing / still lists commands
        return

    def complete(text, state):
        matches = [name for name in COMMANDS if name.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" ")
    if "libedit" in (readline.__doc__ or ""):  # macOS ships libedit, not GNU readline
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def handle_command(command, system, messages):
    """Run a slash command. Returns False when the harness should exit."""
    if command not in COMMANDS:
        # "/" alone, or a partial like "/co": show the commands it could be.
        if not list_commands(command):
            print(f"  unknown command {command}, available commands:")
            list_commands()
        return True

    if command == "/exit":
        return False
    if command == "/help":
        list_commands()
    elif command == "/tokens":
        print_stats()
    elif command == "/compact":
        compact(system, messages)
    elif command == "/clear":
        messages.clear()
        stats["context"] = 0
        print("  conversation cleared")
    elif command == "/save":
        save_session(messages)
    return True


def main():
    system = build_system_prompt()
    messages = []
    enable_tab_completion()
    print("V6 harness - type / for commands (Tab completes), Ctrl+C to interrupt the agent\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            try:
                if not handle_command(user_input, system, messages):
                    break
            except (anthropic.APIError, KeyboardInterrupt) as e:
                print(f"\n  [error] {e or 'interrupted'}")
            continue

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            agent_turn(system, messages)
            if stats["context"] > COMPACT_AT:
                compact(system, messages)
        except KeyboardInterrupt:
            print("\n  [interrupted] request cancelled")
            del messages[checkpoint:]
        except (anthropic.APIError, RuntimeError) as e:
            print(f"\n  [error] {e}")
            del messages[checkpoint:]


if __name__ == "__main__":
    main()
