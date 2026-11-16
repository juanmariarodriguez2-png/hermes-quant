# Working notes for this repository

Hermes Quant trades a real-money account every night from GitHub Actions. Read `README.md` and `docs/GUIA.md` first.

## How work is done here

- **OpenAI Codex takes part in every change.** The owner asked for this (16 Nov 2026). Either Codex writes the change from a
  written brief (`codex exec -m gpt-6.1-sol -c model_reasoning_effort="high" -s workspace-write "<brief>"`, in a separate git
  worktree) and it is then read line by line, or the change is written here and Codex reviews it adversarially before it
  reaches `main` (`-s read-only`). Say in the summary to the owner what Codex did. Codex spends the owner's OpenAI credit:
  use the cheaper model for routine work and tell the owner if the credit runs out.
- `python -m pytest tests -q` must pass before anything is pushed to `main`.
- The nightly workflow commits to `main` (state, prices, `site/live.json`) at about 01:23, 02:53, 04:23 UTC and on weekdays at
  14:45 and 15:20 UTC. Pull before pushing and avoid pushing code at those times.

## Money and secrets

- Never ask for, print or store API keys. They live in GitHub Actions secrets; the owner puts them there himself.
- Switching real trading on or off (`HERMES_MODE`, `HERMES_LIVE_ARMED`, `HERMES_FAVOURITES`, `HERMES_VENUE`, `T212_ENV`) is
  the owner's action, in the repository's variables. Do not arm anything for him.
- The Kraken key must never carry withdrawal permission.
- With a switch off, the daily run must write the same state and orders as before the feature existed. Check it, do not assume it.
- The model is frozen on the design sample (see `docs/RESEARCH.md`). A change to what it buys is a research question first.
