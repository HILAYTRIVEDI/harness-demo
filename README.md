# Build Your Own Agent Harness (from scratch, in Python)

A coding agent is a model plus a **harness**: the loop, the tools, the prompt,
the safety checks and the memory around the model. This repo builds one from
scratch, one layer per branch. It uses no agent framework, only the official
`anthropic` SDK, pointed at [AIML API](https://aimlapi.com), which serves Claude through
the same Messages API format.

| Branch | What it adds | Files |
|--------|--------------|-------|
| `v1` | **A single message**: one API call, one reply, nothing else | `harness.py`, `config.py` |
| `v2` | **The chat loop**: conversation history, because the model is stateless | `harness.py` |
| `v3` | **Tools + the agent loop**: the model asks, the harness executes, results go back | `+ tools.py` |
| `v4` | **System prompt + streaming**: tell the agent who/where it is; print tokens live | `+ prompts.py` |
| `v5` | **Safety, reliability & memory**: permission prompts, sandboxed paths, timeouts, turn limits, error handling, a persistent `MEMORY.md` file | `+ memory.py` |
| `v6` | **Context management**: token tracking, auto-compaction, slash commands, saved sessions | all |

Check out the branches in order (`git checkout v1`, then `v2`, and so on) and
use `git diff v1 v2` to see exactly what each step adds. `main` holds the
finished harness (same as `v6`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your AIML API key in .env
python harness.py
```

This is a demo harness made for teaching, not for production use.
