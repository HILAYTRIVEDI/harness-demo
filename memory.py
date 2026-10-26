"""
Memory: what the agent REMEMBERS between sessions.

The model forgets everything when the harness exits; the conversation history
lives only in RAM. Persistent memory is just a plain file on disk:
  1. at startup the harness reads MEMORY.md into the system prompt
  2. the model calls the `remember` tool to append a new fact to it
  3. next session, that fact is in the system prompt again

No database, no embeddings, no server: an offline Markdown file that you can
open, edit or delete yourself. CLAUDE.md in Claude Code works the same way.
"""

import datetime
from pathlib import Path

MEMORY_FILE = Path.cwd().resolve() / "MEMORY.md"
MAX_MEMORY_CHARS = 10_000  # memory is resent on every call, so keep it small

HEADER = "# Agent memory\n\nFacts the agent saved for future sessions. Edit freely.\n\n"


def load_memory():
    """Return the memory file's text, or '' if nothing has been saved yet."""
    if not MEMORY_FILE.exists():
        return ""
    text = MEMORY_FILE.read_text().strip()
    if len(text) > MAX_MEMORY_CHARS:
        text = text[:MAX_MEMORY_CHARS] + "\n... [memory truncated]"
    return text


def remember(fact):
    """Append one fact to the memory file, with today's date."""
    fact = " ".join(fact.split())  # one fact = one line
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(HEADER)
    with MEMORY_FILE.open("a") as f:
        f.write(f"- {fact} ({datetime.date.today().isoformat()})\n")
    return f"Saved to memory: {fact}"
