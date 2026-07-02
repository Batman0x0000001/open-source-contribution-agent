# Open Source Contribution Agent Harness

This project is a staged Python CLI implementation of an open source contribution agent harness.

Current stage: **s01 - minimal agent loop + bash**.

## What s01 includes

- `osc-agent --repo <path>` interactive CLI.
- `.env` loading for Anthropic-compatible settings.
- A minimal Anthropic Messages-style agent loop.
- A single `bash` tool executed with the target repo as its working directory.
- Command output truncation to 50,000 characters, with a 200-character preview printed during execution.

Later stages will add file tools, permissions, hooks, todo planning, context compaction, memory, task systems, and worktree isolation.

## Configuration

Copy `.env.example` to `.env` and fill in your provider values:

```env
ANTHROPIC_API_KEY=
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
MODEL_ID=deepseek-v4-pro
FALLBACK_MODEL_ID=
```

## Run

```sh
osc-agent --repo /path/to/repo
```

For local development:

```sh
python -m osc_agent.cli --repo /path/to/repo
```

## Verify

```sh
python -m pytest tests
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py
```
