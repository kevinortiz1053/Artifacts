#!/usr/bin/env python3
"""
Goal Tracker + Accountability Check-ins
----------------------------------------
Tracks personal goals (with an optional plan/milestones) and periodically
emails a check-in nudge, the same way chores.py emails chore reminders.

This module is deliberately independent of chores.py -- same philosophy as
the rest of this project: each domain is a self-contained script with its
own data file and its own load/save helpers, runnable standalone via its
own CLI, with no LLM/network dependency baked into the data layer itself.

Where this DOES differ from chores.py: goals are expected to always get an
LLM-authored plan when created (see orchestrator.py's goals_node), so this
module exposes `set_plan()` for that node to call after it drafts one --
but goals.py itself never calls an LLM. It's pure data plumbing, same as
chores.py's load_data/save_data/build_*_from_fields functions.

Data is stored in goals.json (created automatically on first run), next to
this script. Email settings are shared with chores.py via config.json.

Usage:
  python3 goals.py add --name "..." --assignee "..." [--description "..."] [--target MM-DD-YYYY]
  python3 goals.py list [--status active|completed|abandoned] [--assignee "..."]
  python3 goals.py plan <id> --text "..." --milestones "Step 1, Step 2, Step 3"
  python3 goals.py milestone-done <id> <milestone_number>
  python3 goals.py checkin <id> --note "..."
  python3 goals.py complete <id>
  python3 goals.py abandon <id>
  python3 goals.py remove <id>
  python3 goals.py remind      # proactive accountability check-in email
"""

import json
import os
import sys
import argparse
import smtplib
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "goals.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

AI_TAG = "[AI generated]"

# How often (in days) a still-active goal gets a proactive check-in email,
# measured from the later of "last time we sent a check-in" or "when the
# goal was created". Kept as a module constant rather than per-goal so it's
# one obvious knob to tune, same spirit as chores.py's 3-day DUE window.
CHECKIN_INTERVAL_DAYS = 7

VALID_STATUSES = ("active", "completed", "abandoned")


def tag_ai_generated(subject):
    if AI_TAG.lower() not in subject.lower():
        return f"{subject} {AI_TAG}"
    return subject


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"goals": [], "next_id": 1}
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Missing {CONFIG_FILE}. Copy config.json.example to config.json and fill it in.")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def today():
    return date.today()


def parse_flexible_date(s):
    """Accept either MM-DD-YYYY or YYYY-MM-DD. Same convention as chores.py,
    duplicated here rather than imported so this module has no dependency
    on chores.py."""
    s = s.strip()
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"couldn't understand date '{s}' -- use MM-DD-YYYY or YYYY-MM-DD")


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def format_date_mdy(d):
    return d.strftime("%m-%d-%Y")


# --------------------------------------------------------------------------
# Goal CRUD
# --------------------------------------------------------------------------

def build_goal_from_fields(data, fields):
    """Create a goal from a dict of string fields. Appends it to `data` and
    returns the new goal dict, or raises ValueError with a human-readable
    message if something required is missing or invalid.

    Fields: name (required), assignee (required -- who the goal belongs
    to, same household-member concept chores.py uses), description
    (optional), target (optional, MM-DD-YYYY or YYYY-MM-DD),
    checkin_interval (optional, whole number of days between proactive
    check-in emails for this goal -- if omitted, the goal uses the global
    CHECKIN_INTERVAL_DAYS default)."""
    name = (fields.get("name") or "").strip()
    assignee = (fields.get("assignee") or "").strip()

    missing = [k for k, v in [("name", name), ("assignee", assignee)] if not v]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")

    description = (fields.get("description") or "").strip()

    target_raw = fields.get("target") or fields.get("target_date")
    target_date = parse_flexible_date(target_raw).isoformat() if target_raw else None

    checkin_interval = None
    interval_raw = fields.get("checkin_interval") or fields.get("checkin_days")
    if interval_raw:
        try:
            checkin_interval = int(interval_raw)
        except (TypeError, ValueError):
            raise ValueError("checkin_interval must be a whole number of days")
        if checkin_interval < 1:
            raise ValueError("checkin_interval must be at least 1 day")

    goal = {
        "id": data["next_id"],
        "name": name,
        "assignee": assignee,
        "description": description,
        "target_date": target_date,
        "status": "active",
        "plan": None,
        "milestones": [],
        "checkins": [],
        "created": today().isoformat(),
        "last_checkin_sent": None,
        "checkin_interval_days": checkin_interval,
    }
    data["goals"].append(goal)
    data["next_id"] += 1
    return goal


def find_goal(data, goal_id):
    return next((g for g in data["goals"] if g["id"] == goal_id), None)


def set_plan(data, goal_id, plan_text, milestones=None):
    """Attach (or replace) a plan and its milestone list on a goal. Meant to
    be called right after build_goal_from_fields() by whoever drafted the
    plan (the goals LLM agent in orchestrator.py, or a human via the CLI).
    `milestones` is a list of plain milestone description strings; each
    becomes {"text": ..., "done": False}. Returns the goal, or None if no
    goal has that id."""
    goal = find_goal(data, goal_id)
    if goal is None:
        return None
    goal["plan"] = plan_text
    if milestones is not None:
        goal["milestones"] = [{"text": m.strip(), "done": False} for m in milestones if m.strip()]
    return goal


def set_checkin_interval(data, goal_id, days):
    """Set (or clear, with days=None) how often -- in days -- this specific
    goal gets a proactive check-in email, overriding the global
    CHECKIN_INTERVAL_DAYS default. Returns the goal, or None if no goal has
    that id. Raises ValueError if `days` isn't a positive whole number."""
    goal = find_goal(data, goal_id)
    if goal is None:
        return None
    if days is not None:
        days = int(days)
        if days < 1:
            raise ValueError("checkin interval must be at least 1 day")
    goal["checkin_interval_days"] = days
    return goal


def mark_milestone_done(data, goal_id, milestone_number):
    """milestone_number is 1-based, matching what build_status_report()
    displays. Returns (goal, milestone_text) on success, or (None, None) if
    the goal or milestone number doesn't exist."""
    goal = find_goal(data, goal_id)
    if goal is None:
        return None, None
    idx = milestone_number - 1
    if idx < 0 or idx >= len(goal["milestones"]):
        return None, None
    goal["milestones"][idx]["done"] = True
    return goal, goal["milestones"][idx]["text"]


def log_checkin(data, goal_id, note):
    """Record a check-in note against a goal -- this both logs progress and
    resets the accountability clock (see get_days_since_checkin)."""
    goal = find_goal(data, goal_id)
    if goal is None:
        return None
    goal["checkins"].append({"date": today().isoformat(), "note": note.strip()})
    return goal


def set_status(data, goal_id, status):
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(VALID_STATUSES)}")
    goal = find_goal(data, goal_id)
    if goal is None:
        return None
    goal["status"] = status
    return goal


def remove_goals(data, goal_ids):
    """Remove multiple goals by id in one atomic step, same pattern as
    chores.py's remove_chores: resolve every id before removing/renumbering
    anything. Returns (removed_goals, not_found_ids)."""
    to_remove, not_found, seen = [], [], set()
    for goal_id in goal_ids:
        if goal_id in seen:
            continue
        seen.add(goal_id)
        match = find_goal(data, goal_id)
        (to_remove if match else not_found).append(match or goal_id)

    for g in to_remove:
        data["goals"].remove(g)

    if to_remove:
        for new_id, g in enumerate(data["goals"], start=1):
            g["id"] = new_id
        data["next_id"] = len(data["goals"]) + 1

    return to_remove, not_found


def remove_goal(data, goal_id):
    removed, _ = remove_goals(data, [goal_id])
    return removed[0] if removed else None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def describe_progress(goal):
    if not goal["milestones"]:
        return "no plan yet" if goal["plan"] is None else "plan set, no milestones listed"
    done = sum(1 for m in goal["milestones"] if m["done"])
    total = len(goal["milestones"])
    return f"{done}/{total} milestones done"


def next_incomplete_milestone(goal):
    for m in goal["milestones"]:
        if not m["done"]:
            return m["text"]
    return None


def format_goal_line(g):
    target = f", target {format_date_mdy(parse_date(g['target_date']))}" if g.get("target_date") else ""
    next_step = next_incomplete_milestone(g)
    next_str = f" | next: {next_step}" if next_step else ""
    return (f"#{g['id']:<3} [{g['status']:<9}] {g['name']:<30} -> {g.get('assignee', '?'):<10} "
            f"({describe_progress(g)}{target}){next_str}")


def build_status_report(data, status_filter=None, assignee=None):
    """Plain-text status listing. If `assignee` is given, only that
    person's goals are included (case-insensitive, exact match on the
    assignee field) -- used both for an explicit 'list Alice'-style
    request and, more commonly now, as the default scoping applied by
    orchestrator.py's goals_node so a household member's email only shows
    their own goals unless they explicitly ask for everyone's."""
    goals = data["goals"]
    if assignee:
        goals = [g for g in goals if g.get("assignee", "").strip().lower() == assignee.strip().lower()]
        if not goals:
            return f"No goals found for '{assignee}'."
    if status_filter:
        goals = [g for g in goals if g["status"] == status_filter]
        if not goals:
            return f"No {status_filter} goals found" + (f" for '{assignee}'." if assignee else ".")

    if not goals:
        return "No goals yet."

    active = [g for g in goals if g["status"] == "active"]
    other = [g for g in goals if g["status"] != "active"]

    header = f"Current goals for {assignee}:" if assignee else "Current goals:"
    lines = [header, ""]
    if active:
        lines.append("ACTIVE:")
        for g in active:
            lines.append(format_goal_line(g))
    if other:
        if active:
            lines.append("")
        lines.append("OTHER:")
        for g in other:
            lines.append(format_goal_line(g))

    lines.append("")
    lines.append("Reply 'milestone done <goal id> <number>' to check off a step, "
                 "or 'goal checkin <id>: <note>' to log progress.")
    return "\n".join(lines)


def build_goal_detail(goal):
    """Full detail view for one goal -- used when someone asks about a
    specific goal by name/id rather than the whole list."""
    lines = [f"#{goal['id']} {goal['name']} [{goal['status']}] -- {goal.get('assignee', '?')}"]
    if goal["description"]:
        lines.append(goal["description"])
    if goal.get("target_date"):
        lines.append(f"Target: {format_date_mdy(parse_date(goal['target_date']))}")
    if goal.get("checkin_interval_days"):
        lines.append(f"Check-in frequency: every {goal['checkin_interval_days']} day(s) (custom)")
    lines.append("")
    if goal["plan"]:
        lines.append("Plan:")
        lines.append(goal["plan"])
        lines.append("")
    if goal["milestones"]:
        lines.append("Milestones:")
        for i, m in enumerate(goal["milestones"], start=1):
            mark = "x" if m["done"] else " "
            lines.append(f"  [{mark}] {i}. {m['text']}")
        lines.append("")
    if goal["checkins"]:
        lines.append("Recent check-ins:")
        for c in goal["checkins"][-5:]:
            lines.append(f"  {c['date']}: {c['note']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Email (shared config.json with chores.py, same SMTP pattern)
# --------------------------------------------------------------------------

def send_email(subject, body, config, to_list=None):
    """Send a plain (non-reply) automated email. Defaults to broadcasting
    to every address in config["recipients"] -- pass an explicit `to_list`
    to send to specific address(es) instead (see cmd_remind's per-assignee
    routing below). Same pattern as chores.py's send_email."""
    to_list = to_list if to_list is not None else config["recipients"]
    msg = MIMEText(body)
    msg["Subject"] = tag_ai_generated(subject)
    msg["From"] = config["sender_email"]
    msg["To"] = ", ".join(to_list)

    with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
        server.starttls()
        server.login(config["sender_email"], config["sender_password"])
        server.sendmail(config["sender_email"], to_list, msg.as_string())


def get_assignee_email(config, assignee):
    """Look up an assignee's email address from config["assignee_emails"]
    (the SAME {name: email} directory chores.py uses -- shared via
    config.json, matched case-insensitively). Returns None if there's no
    entry."""
    directory = config.get("assignee_emails") or {}
    lookup = {name.strip().lower(): addr for name, addr in directory.items()}
    return lookup.get(assignee.strip().lower())


def effective_checkin_interval(goal):
    """The number of days between proactive check-ins for this goal: its
    own override if one was set (via 'checkin_interval' on creation, or
    set_checkin_interval() later), otherwise the global default."""
    return goal.get("checkin_interval_days") or CHECKIN_INTERVAL_DAYS


def get_days_since_checkin(goal, on_day):
    """Days since the later of: the last time we SENT a proactive check-in
    email, or (if none sent yet) when the goal was created. Note this is
    about outbound nudges, not about goal["checkins"] (which log the
    person's replies/progress) -- a goal that gets checked in on organically
    via email still gets nudged again on the normal interval; the two are
    tracked separately on purpose so an active conversation about a goal
    doesn't suppress the periodic accountability email."""
    anchor = parse_date(goal["last_checkin_sent"]) if goal.get("last_checkin_sent") else parse_date(goal["created"])
    return (on_day - anchor).days


def build_checkin_email_body(due_goals):
    lines = ["Goal check-in", ""]
    for g in due_goals:
        lines.append(f"#{g['id']} {g['name']} ({describe_progress(g)})")
        next_step = next_incomplete_milestone(g)
        if next_step:
            lines.append(f"  Next step: {next_step}")
        if g.get("target_date"):
            lines.append(f"  Target: {format_date_mdy(parse_date(g['target_date']))}")
        lines.append("")
    lines.append("Reply with an update on any of these, or 'milestone done <id> <number>' "
                 "to check off a completed step.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_add(args):
    data = load_data()
    fields = {
        "name": args.name,
        "assignee": args.assignee,
        "description": args.description,
        "target": args.target,
        "checkin_interval": str(args.checkin_days) if args.checkin_days is not None else None,
    }
    try:
        goal = build_goal_from_fields(data, fields)
    except ValueError as e:
        print(f"Couldn't add goal: {e}")
        sys.exit(1)
    save_data(data)
    print(f"Added goal #{goal['id']}: {goal['name']}")


def cmd_checkin_frequency(args):
    data = load_data()
    try:
        goal = set_checkin_interval(data, args.id, args.days)
    except ValueError as e:
        print(f"Couldn't set check-in frequency: {e}")
        sys.exit(1)
    if goal:
        save_data(data)
        if args.days is None:
            print(f"Reset #{goal['id']} '{goal['name']}' to the default check-in frequency "
                  f"({CHECKIN_INTERVAL_DAYS} days).")
        else:
            print(f"#{goal['id']} '{goal['name']}' will now get a check-in every {args.days} day(s).")
    else:
        print(f"No goal with id {args.id}")


def cmd_list(args):
    data = load_data()
    print(build_status_report(data, status_filter=args.status, assignee=args.assignee))


def cmd_plan(args):
    data = load_data()
    milestones = [m for m in (args.milestones or "").split(",")] if args.milestones else None
    goal = set_plan(data, args.id, args.text, milestones)
    if goal:
        save_data(data)
        print(f"Set plan for #{goal['id']} '{goal['name']}'.")
    else:
        print(f"No goal with id {args.id}")


def cmd_milestone_done(args):
    data = load_data()
    goal, text = mark_milestone_done(data, args.id, args.number)
    if goal:
        save_data(data)
        print(f"Marked milestone {args.number} done for #{goal['id']} '{goal['name']}': {text}")
    else:
        print(f"No goal/milestone found for goal {args.id}, milestone {args.number}")


def cmd_checkin(args):
    data = load_data()
    goal = log_checkin(data, args.id, args.note)
    if goal:
        save_data(data)
        print(f"Logged check-in for #{goal['id']} '{goal['name']}'.")
    else:
        print(f"No goal with id {args.id}")


def cmd_complete(args):
    data = load_data()
    goal = set_status(data, args.id, "completed")
    if goal:
        save_data(data)
        print(f"Marked #{goal['id']} '{goal['name']}' completed. Nice work!")
    else:
        print(f"No goal with id {args.id}")


def cmd_abandon(args):
    data = load_data()
    goal = set_status(data, args.id, "abandoned")
    if goal:
        save_data(data)
        print(f"Marked #{goal['id']} '{goal['name']}' abandoned.")
    else:
        print(f"No goal with id {args.id}")


def cmd_remove(args):
    data = load_data()
    g = remove_goal(data, args.id)
    if g:
        save_data(data)
        print(f"Removed goal #{args.id}")
    else:
        print(f"No goal with id {args.id}")


def cmd_remind(args):
    data = load_data()
    config = load_config()

    on_day = today()
    active = [g for g in data["goals"] if g["status"] == "active"]
    due = [g for g in active if get_days_since_checkin(g, on_day) >= effective_checkin_interval(g)]

    if not due:
        print("No goals due for a check-in today.")
        return

    if not config.get("assignee_emails"):
        # Legacy broadcast behavior, unchanged: no directory configured, so
        # send everything to config["recipients"] same as before.
        body = build_checkin_email_body(due)
        subject = f"Goal check-in - {len(due)} goal(s)"
        send_email(subject, body, config)
        for g in due:
            g["last_checkin_sent"] = on_day.isoformat()
        save_data(data)
        print("Check-in email sent.")
        print(body)
        return

    # Per-assignee routing, same pattern as chores.py's cmd_remind.
    assignees_needing_checkin = sorted({g.get("assignee", "") for g in due if g.get("assignee")})
    unmapped = []
    for assignee in assignees_needing_checkin:
        a_due = [g for g in due if g.get("assignee") == assignee]
        body = build_checkin_email_body(a_due)
        subject = f"Goal check-in - {len(a_due)} goal(s)"
        to_addr = get_assignee_email(config, assignee)
        if to_addr:
            send_email(subject, body, config, to_list=[to_addr])
            print(f"Check-in sent to {assignee} <{to_addr}>.")
        else:
            unmapped.append((assignee, body))

    if unmapped:
        names = ", ".join(a for a, _ in unmapped)
        print(f"No email on file for: {names}. Add them to config.json's "
              f"'assignee_emails' -- routing their check-in(s) to "
              f"{config['sender_email']} for now.")
        combined_body = "\n\n---\n\n".join(f"[{a}]\n{b}" for a, b in unmapped)
        subject = f"Goal check-in - no email on file for: {names}"
        send_email(subject, combined_body, config, to_list=[config["sender_email"]])

    for g in due:
        g["last_checkin_sent"] = on_day.isoformat()
    save_data(data)

    print("Check-in email(s) sent.")


def main():
    parser = argparse.ArgumentParser(description="Goal tracker with accountability check-ins")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new goal")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--assignee", required=True, help="Who this goal belongs to")
    p_add.add_argument("--description", default="")
    p_add.add_argument("--target", help="MM-DD-YYYY or YYYY-MM-DD, optional")
    p_add.add_argument("--checkin-days", dest="checkin_days", type=int,
                        help=f"Days between proactive check-ins for this goal "
                             f"(default: {CHECKIN_INTERVAL_DAYS})")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List goals and their progress")
    p_list.add_argument("--status", choices=VALID_STATUSES)
    p_list.add_argument("--assignee", help="Only show goals for this person")
    p_list.set_defaults(func=cmd_list)

    p_plan = sub.add_parser("plan", help="Set/replace a goal's plan and milestones")
    p_plan.add_argument("id", type=int)
    p_plan.add_argument("--text", required=True)
    p_plan.add_argument("--milestones", help="Comma-separated milestone descriptions")
    p_plan.set_defaults(func=cmd_plan)

    p_ms = sub.add_parser("milestone-done", help="Mark a milestone complete")
    p_ms.add_argument("id", type=int)
    p_ms.add_argument("number", type=int, help="1-based milestone number, from 'list'/plan view")
    p_ms.set_defaults(func=cmd_milestone_done)

    p_checkin = sub.add_parser("checkin", help="Log a check-in note against a goal")
    p_checkin.add_argument("id", type=int)
    p_checkin.add_argument("--note", required=True)
    p_checkin.set_defaults(func=cmd_checkin)

    p_complete = sub.add_parser("complete", help="Mark a goal completed")
    p_complete.add_argument("id", type=int)
    p_complete.set_defaults(func=cmd_complete)

    p_abandon = sub.add_parser("abandon", help="Mark a goal abandoned")
    p_abandon.add_argument("id", type=int)
    p_abandon.set_defaults(func=cmd_abandon)

    p_remove = sub.add_parser("remove", help="Delete a goal entirely")
    p_remove.add_argument("id", type=int)
    p_remove.set_defaults(func=cmd_remove)

    p_remind = sub.add_parser("remind", help="Email a proactive check-in for goals due one")
    p_remind.set_defaults(func=cmd_remind)

    p_freq = sub.add_parser("checkin-frequency", help="Set (or reset) how often a goal gets checked in on")
    p_freq.add_argument("id", type=int)
    p_freq.add_argument("days", type=int, nargs="?", default=None,
                         help=f"Days between check-ins; omit to reset to the default ({CHECKIN_INTERVAL_DAYS})")
    p_freq.set_defaults(func=cmd_checkin_frequency)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
