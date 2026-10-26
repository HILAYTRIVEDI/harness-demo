"""
Config: which API and model the harness talks to.

Settings come from environment variables. For convenience they can also live
in a .env file next to this one (copy .env.example to .env and fill it in).
"""

import os
import sys
from pathlib import Path

import anthropic


def load_env_file(path=Path(__file__).parent / ".env"):
    """Tiny .env loader: KEY=value lines. Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

API_KEY = os.environ.get("AIML_API_KEY")
if not API_KEY:
    sys.exit("AIML_API_KEY is not set. Put it in a .env file (see .env.example) or export it.")

MODEL = os.environ.get("AIML_MODEL", "claude-opus-4-6")

# AIML API serves Claude through the same Messages API format as Anthropic,
# so the official SDK works unchanged: we only swap the base URL and the key.
client = anthropic.Anthropic(base_url="https://api.aimlapi.com", api_key=API_KEY)
