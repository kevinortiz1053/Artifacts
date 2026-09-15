# Household Chief of Staff

An email-only AI agent that tracks household chores and personal goals, sends its own reminders, and understands plain-language requests — no app, no login, just email.

Two things power it:

- **`chores.py`** and **`goals.py`** — self-contained, dependency-free trackers with their own CLIs, JSON storage, and a fast regex layer that handles common commands (`list`, `done 3`, `add …`) without ever calling an LLM.
- **`orchestrator.py`** — a [LangGraph](https://github.com/langchain-ai/langgraph) multi-agent pipeline that kicks in for anything the regex layer can't confidently parse: natural language, multi-part requests ("mark the trash done *and* update my running goal"), and new goal planning.

If you never configure an Anthropic API key, the LLM layer is never touched — `chores.py` still works standalone, just with plainer replies and no free-text understanding.

## How it works, from the inbox

1. Email the assistant with subject `chores` (or reply to any thread it started).
2. `chores.py check-email` (run on a schedule) picks it up, and `route_message()` decides how to handle it:
   - A **pending confirmation** (yes/no on a proposed deletion or multi-step plan) is resolved first, outside the LLM entirely.
   - A plain command the regex layer recognizes (`list`, `status`, `done <n>`, `remove <n>`, structured `add`) is handled immediately, for free.
   - Anything else — if an API key is configured — goes to `orchestrator.py`.
3. Inside the orchestrator, a **Supervisor** (Haiku) decomposes the message into one or more steps, each assigned to a specialist:
   - **Chores** — tries the same regex fast path first, then falls back to a tool-using agent (Sonnet) for anything freeform.
   - **Goals** — always a tool-using agent; every new goal needs an LLM-drafted plan and milestones. Tools are scoped to the sender, so by default you only see/manage your own goals.
   - **General** — read-only lookups across both domains. It has no write tools at all (not a prompt restriction — they simply aren't in its tool list), so it can't be talked into modifying anything.
   - Multi-step requests are staged as a pending plan and emailed back as a preview for confirmation before anything runs; a single-step request runs immediately.
   - After each specialist step, control returns to the Supervisor, which judges whether the step actually completed and can send it back for another attempt (up to 4 hops) before moving on.
4. A final **Vibe** pass (Sonnet) rewrites the combined results into one warm, concise reply — without inventing or dropping any concrete fact (chore/goal names, IDs, dates, milestones).
5. The reply lands in the same email thread via IMAP/SMTP, using the original message's `In-Reply-To`/`References` headers.

Proactive reminders (daily chore digest, goal check-ins) are sent separately via `chores.py remind` / `goals.py remind`, each starting a **fresh** thread rather than replying in-thread.

## Files

| File | Purpose |
|---|---|
| `chores.py` | Chore tracker: CLI (`add`/`list`/`done`/`remove`/`remind`/`check-email`), due-date/recurrence logic, email send/receive, regex command parser, confirm-before-delete flow, and the routing entry point into the orchestrator. |
| `goals.py` | Goal tracker: CLI (`add`/`list`/`plan`/`milestone-done`/`checkin`/`complete`/`abandon`/`remove`/`remind`/`checkin-frequency`), plans/milestones, check-in cadence and logging. |
| `orchestrator.py` | The LangGraph pipeline (Supervisor → Chores/Goals/General → Supervisor → Vibe) described above. Only imported/used when an Anthropic API key is configured; otherwise `chores.py` degrades gracefully to regex + help text. |
| `inspect_chores_data.py` | Diagnostic script — prints exactly which `chores.json` a given run of `chores.py` is reading (path, contents, last modified) and scans disk for stray duplicate copies. Useful when "my chore disappeared" turns out to be two different working directories. |
| `test_llm_routing.py` | Lets you test, offline, whether a given email body would be caught by the regex layer or handed to the LLM orchestrator, without touching your real inbox or `chores.json` (unless you pass `--live` and it genuinely falls through to the LLM). |
| `requirements.txt` | Only needed for the LLM orchestrator feature — `langgraph`, `langchain-core`, `langchain-anthropic`. `chores.py`/`goals.py` run fine without them. |
| `watch_inbox.py` *(not in this drop)* | Long-running alternative to cron'ing `chores.py check-email` — a persistent process that watches the inbox continuously instead of polling once per invocation. `orchestrator.py` already accounts for it: LLM graphs are cached per API key at module level specifically so a long-running watcher doesn't rebuild the LangGraph pipeline on every single email. Add it to this table (with its own row above, matching the others) once it's included in the repo. |
| `Household_Chief_of_Staff_User_Guide.pdf` | End-user guide — what to say and how the assistant responds. Written for household members, not developers. |

## Setup

### 1. Install dependencies

```bash
# Core (chore/goal tracking + email, no LLM)
python3 -m venv venv && source venv/bin/activate

# Optional: enables natural-language / multi-step requests
pip install -r requirements.txt
```

### 2. Configure email + (optionally) the LLM

Create `config.json` next to `chores.py`:

```json
{
  "sender_email": "your-bot@example.com",
  "sender_password": "an app password, not your real password",
  "smtp_host": "smtp.example.com",
  "smtp_port": 587,
  "imap_host": "imap.example.com",
  "imap_port": 993,
  "recipients": ["you@example.com"],
  "assignee_emails": {
    "Alice": "alice@example.com",
    "Bob": "bob@example.com"
  },
  "anthropic_api_key": "sk-ant-..."
}
```

Notes:
- `imap_host`/`imap_port` are optional — if omitted, `imap_host` is guessed by swapping `smtp` → `imap` in `smtp_host`.
- `assignee_emails` is optional. Without it, everything broadcasts to `recipients`; with it, per-person reminders and orchestrator goal-scoping ("only my own goals") both become possible.
- `anthropic_api_key` can also come from the `ANTHROPIC_API_KEY` environment variable. Leave both unset to run without the LLM orchestrator at all.
- Only senders listed in `recipients` or `assignee_emails` get a response — anyone else's email is ignored.

### 3. Run it

```bash
# One-off / manual
python3 chores.py add --name "Take out trash" --assignee Alice --type recurring --frequency daily
python3 chores.py list
python3 goals.py add --assignee Bob --description "Run a 5k by spring"

# Scheduled (cron / Task Scheduler)
python3 chores.py remind        # daily chore digest
python3 goals.py remind         # goal check-ins that are due
python3 chores.py check-email   # poll inbox once, reply to anything new

# Or, instead of cron'ing check-email: a long-running watcher (watch_inbox.py,
# not included in this drop) that stays connected and picks up new mail as it
# arrives, rather than polling once per invocation.
```

Run `check-email` on whatever cadence you want incoming email checked (e.g. every few minutes via cron), and `remind` once a day — or run `watch_inbox.py` once as a persistent process if you'd rather not cron the polling at all.

### 4. Debugging tools

```bash
# Which chores.json is check-email actually reading? Any stray duplicates on disk?
python3 inspect_chores_data.py

# Would this message be handled by regex, or routed to the LLM? (dry run by default)
python3 test_llm_routing.py "can you mow the lawn every two weeks for Bob?"
python3 test_llm_routing.py --live "what am I behind on this week?"   # actually calls the LLM
```

## Design notes worth knowing

- **Loop prevention** doesn't rely on subject-line tagging (which gets echoed back on Reply) — it uses a dedicated `X-Chore-Bot` header, so the bot can't accidentally reply to itself.
- **Deletions are always confirm-first**: a plain-language delete request is staged and previewed by email; nothing is removed until you reply "yes". Confirmations expire after 24 hours.
- **Multi-step requests get the same treatment**: the Supervisor's plan is emailed back as a preview before any specialist is actually invoked, so a bad split can be corrected (or cancelled) before it does anything.
- **The General specialist is structurally read-only** — it has no write tools in its tool list at all, not just a prompt telling it not to write, so it can't be turned into a backdoor write path even as part of a larger plan.
- **Regex stays the fast/free path everywhere it can**: both `chores.py`'s standalone mode and the orchestrator's Chores node try it first, so plain `done 3`/`list`/`add ...` commands never cost an API call.

## Requirements

- Python 3
- An SMTP/IMAP-capable email account for the bot to send/receive from
- Optional: an Anthropic API key + `pip install -r requirements.txt`, for natural-language and multi-step request handling
