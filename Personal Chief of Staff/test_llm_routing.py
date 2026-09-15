#!/usr/bin/env python3
"""
Test whether a given email body would be handled locally (regex) or handed
off to the LLM orchestrator -- without touching your real inbox or
chores.json.

Usage:
    python3 test_llm_routing.py                 # run the built-in examples
    python3 test_llm_routing.py "your message"   # test one message

Place this file in the same folder as chores.py before running.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chores  # noqa: E402


def check_regex_only(body):
    """Step 1: would parse_command() catch this locally, or fall through?"""
    command, arg = chores.parse_command(body)
    if command is None:
        return None
    return f"{command} {arg}"


def check_full_pipeline(body, sender="test@example.com"):
    """Step 2: run it through handle_incoming_email() with a throwaway,
    in-memory chore list -- never touches your real chores.json for the
    regex-handled commands. NOTE: if this message falls through to the LLM
    orchestrator, its tools (add_chore/list_chores/mark_done in
    orchestrator.py) call chore_lib.load_data()/save_data() directly, which
    *does* read/write your real chores.json. Back it up first if you test
    this path."""
    data = {"chores": [], "next_id": 1}
    reply, changed = chores.handle_incoming_email(body, data, sender=sender)
    return reply


def explain(body):
    print(f"\n--- message: {body!r} ---")
    local = check_regex_only(body)
    if local is not None:
        print(f"  -> handled locally by regex: {local}")
        print("     (never reaches the LLM)")
        return
    print("  -> regex found nothing (parse_command returned None, None)")

    config = chores.load_config()
    api_key = config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  -> no anthropic_api_key configured (config.json or env var)")
        print("     falls back to plain help text -- LLM never called")
        return

    try:
        import orchestrator  # noqa: F401
    except ImportError as e:
        print(f"  -> orchestrator import failed ({e})")
        print("     run: pip install -r requirements.txt --break-system-packages")
        print("     falls back to plain help text -- LLM never called")
        return

    print("  -> would be routed to the LLM orchestrator (API key + deps present)")
    print("     re-run with --live to actually call it and see the reply")


EXAMPLES = [
    "list",
    "status",
    "done 3",
    "add\nname: Take out trash\nassignee: Alice\ntype: recurring\nfrequency: daily",
    "please add me to your mailing list",   # gotcha: "list" substring matches step 5
    "what am I behind on this week?",
    "can you mow the lawn every two weeks for Bob?",
]


if __name__ == "__main__":
    live = "--live" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--live"]

    messages = args if args else EXAMPLES
    for msg in messages:
        explain(msg)
        if live and check_regex_only(msg) is None:
            print("  --- calling the real pipeline (may cost an API call) ---")
            print(check_full_pipeline(msg))
