"""
Tools: what the agent can DO.

Each tool has two halves:
  1. a SCHEMA (name, description, input_schema) that we send to the model
  2. a FUNCTION that the harness runs when the model asks for that tool

The model never runs anything itself. It only asks for a tool call, and the
harness decides whether and how to run it.
"""

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Tool implementations (plain Python functions)
# ---------------------------------------------------------------------------

def list_files(path="."):
    entries = sorted(Path(path).iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"


def read_file(path):
    return Path(path).read_text()


def write_file(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def edit_file(path, old_text, new_text):
    p = Path(path)
    text = p.read_text()
    count = text.count(old_text)
    if count != 1:
        return f"Error: old_text must appear exactly once in {path}, found {count} matches"
    p.write_text(text.replace(old_text, new_text))
    return f"Edited {path}"


def run_bash(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
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
        "description": "Run a shell command and return its exit code and output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
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
}


def run_tool(name, tool_input):
    return TOOL_FUNCTIONS[name](**tool_input)
