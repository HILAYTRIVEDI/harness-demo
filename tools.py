"""
Tools: what the agent can DO.

Each tool has two halves:
  1. a SCHEMA (name, description, input_schema) that we send to the model
  2. a FUNCTION that the harness runs when the model asks for that tool

V5 adds guard rails, because the model's tool inputs are untrusted:
  - every path is confined to the workspace folder
  - shell commands get a timeout
  - long outputs are truncated so they don't flood the context window
  - errors are returned to the model instead of crashing the harness
It also adds `remember`, which saves a fact to persistent memory (see memory.py).
"""

import subprocess
from pathlib import Path

from memory import remember

WORKSPACE = Path.cwd().resolve()
BASH_TIMEOUT = 60          # seconds
MAX_OUTPUT_CHARS = 20_000  # roughly 5k tokens

# Tools that change things. The harness asks the user before running these.
# remember is here too: whatever it saves is fed into every future session.
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_bash", "remember"}


class ToolError(Exception):
    pass


def safe_path(path):
    """Resolve a path and refuse anything outside the workspace."""
    p = (WORKSPACE / path).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ToolError(f"Path {path} is outside the workspace {WORKSPACE}")
    return p


# ---------------------------------------------------------------------------
# 1. Tool implementations (plain Python functions)
# ---------------------------------------------------------------------------

def list_files(path="."):
    entries = sorted(safe_path(path).iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"


def read_file(path):
    return safe_path(path).read_text()


def write_file(path, content):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def edit_file(path, old_text, new_text):
    p = safe_path(path)
    text = p.read_text()
    count = text.count(old_text)
    if count != 1:
        raise ToolError(f"old_text must appear exactly once in {path}, found {count} matches")
    p.write_text(text.replace(old_text, new_text))
    return f"Edited {path}"


def run_bash(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=WORKSPACE, timeout=BASH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {BASH_TIMEOUT}s")
    return f"exit code: {result.returncode}\n{result.stdout}{result.stderr}"


# ---------------------------------------------------------------------------
# 2. Tool schemas (what the model sees)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_files",
        "description": "List the files and folders in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list. Defaults to '.'"},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read the full text content of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a file or overwrite it completely with new content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace one exact, unique snippet of text in an existing file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to find. Must match once."},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "run_bash",
        "description": f"Run a shell command in the workspace and return its exit code and output. "
                       f"Commands time out after {BASH_TIMEOUT}s; don't start servers or interactive programs.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "remember",
        "description": "Save one short fact to persistent memory so you still know it in future sessions. "
                       "Use it for lasting user preferences and project facts, not for temporary task details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "One self-contained sentence."},
            },
            "required": ["fact"],
        },
    },
]


# ---------------------------------------------------------------------------
# 3. Dispatcher: tool name -> function
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_bash": run_bash,
    "remember": remember,
}


def truncate(text):
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    cut = len(text) - MAX_OUTPUT_CHARS
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated {cut} characters]"


def run_tool(name, tool_input):
    """Run a tool and return (output, is_error). Never raises."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool {name}", True
    try:
        return truncate(TOOL_FUNCTIONS[name](**tool_input)), False
    except Exception as e:  # any failure goes back to the model as an error result
        return f"Error: {type(e).__name__}: {e}", True
