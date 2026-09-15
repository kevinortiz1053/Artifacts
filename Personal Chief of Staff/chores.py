#!/usr/bin/env python3
"""
Household Chore Tracker + Email Reminders
-------------------------------------------
Tracks recurring and ad-hoc chores for a household and emails a reminder
of what's due/overdue. Designed to be run:
  - manually, via subcommands (add / list / done / remove)
  - on a schedule (cron / Task Scheduler) with `python3 chores.py remind`
  - on a schedule with `python3 chores.py check-email` to let people reply
    to the reminder email with "list"/"status" or "done <number>" and get
    an emailed response back.

Data is stored in chores.json (created automatically on first run).
Email settings live in config.json (see config.json.example).

See README.md for full setup instructions.
"""

import json
import os
import re
import smtplib
import imaplib
import email
import email.utils
import sys
import argparse
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "chores.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
EMAIL_STATE_FILE = os.path.join(BASE_DIR, "email_state.json")
PENDING_DELETIONS_FILE = os.path.join(BASE_DIR, "pending_deletions.json")

# How long a proposed-but-unconfirmed deletion stays valid. Past this, a
# stray "yes" showing up days later on an unrelated thread won't delete
# anything -- the person has to ask again and get a fresh confirmation.
PENDING_DELETION_TTL_HOURS = 24

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Every automated email this script sends carries this tag in its subject,
# purely so a human glancing at their inbox can tell it's automated.
AI_TAG = "[AI generated]"

# Real loop-prevention uses a custom header instead of the subject, since
# subjects get echoed back verbatim when someone hits "Reply" — a header
# does not, so it can't be confused with a genuine human reply.
BOT_HEADER = "X-Chore-Bot"


def tag_ai_generated(subject):
    if AI_TAG.lower() not in subject.lower():
        return f"{subject} {AI_TAG}"
    return subject


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"chores": [], "next_id": 1}
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_email_state():
    if not os.path.exists(EMAIL_STATE_FILE):
        return {"last_uid": None}
    with open(EMAIL_STATE_FILE, "r") as f:
        return json.load(f)


def save_email_state(state):
    with open(EMAIL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Missing {CONFIG_FILE}. Copy config.json.example to config.json and fill it in.")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def today():
    return date.today()


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------------------------------------------------------
# Due-date logic
# --------------------------------------------------------------------------


def get_scheduled_start(chore, on_day):
    """Return the most recent date on or before `on_day` that this chore started being due."""
    if chore["type"] == "adhoc":
        return parse_date(chore["due_date"])

    freq = chore["frequency"]

    if freq == "daily":
        return on_day

    if freq == "weekly":
        days_names = chore.get("days", [])
        for delta in range(0, 7):
            candidate = on_day - timedelta(days=delta)
            if candidate.strftime("%A") in days_names:
                return candidate
        return on_day  # no valid days configured; fall back to today

    if freq == "monthly":
        dom = chore.get("day_of_month", 1)
        this_month_first = on_day.replace(day=1)
        # last valid day in the current month (handles short months, e.g. dom=31 in Feb)
        next_month_first = (this_month_first.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day_this_month = (next_month_first - timedelta(days=1)).day
        candidate = on_day.replace(day=min(dom, last_day_this_month))
        if candidate <= on_day:
            return candidate
        prev_month_last_day = this_month_first - timedelta(days=1)
        return prev_month_last_day.replace(day=min(dom, prev_month_last_day.day))

    if freq == "interval":
        anchor = parse_date(chore.get("start_date", on_day.isoformat()))
        interval = chore.get("interval_days", 7)
        if on_day < anchor:
            return anchor
        n = (on_day - anchor).days // interval
        return anchor + timedelta(days=n * interval)

    return on_day


def get_status(chore, on_day):
    """Three-state status: DONE (completed since it started), DUE (started within
    the last 3 days), or OVERDUE (started more than 3 days ago and not done)."""
    start = get_scheduled_start(chore, on_day)
    last_done = parse_date(chore["last_completed"]) if chore.get("last_completed") else None

    if last_done and last_done >= start:
        return "DONE"

    days_since_start = (on_day - start).days
    if days_since_start <= 3:
        return "DUE"
    return "OVERDUE"


def get_next_due(chore, on_day):
    """Return the date this chore will next come due, after its current scheduled
    occurrence (regardless of whether the current one is done/due/overdue).
    Returns None for one-time (adhoc) chores, since they don't recur."""
    if chore["type"] == "adhoc":
        return None

    freq = chore["frequency"]
    current_start = get_scheduled_start(chore, on_day)

    if freq == "daily":
        return current_start + timedelta(days=1)

    if freq == "weekly":
        days_names = chore.get("days", [])
        for delta in range(1, 8):
            candidate = current_start + timedelta(days=delta)
            if candidate.strftime("%A") in days_names:
                return candidate
        return None

    if freq == "monthly":
        dom = chore.get("day_of_month", 1)
        next_month_first = (current_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        following_month_first = (next_month_first.replace(day=28) + timedelta(days=4)).replace(day=1)
        last_day_next_month = (following_month_first - timedelta(days=1)).day
        return next_month_first.replace(day=min(dom, last_day_next_month))

    if freq == "interval":
        interval = chore.get("interval_days", 7)
        return current_start + timedelta(days=interval)

    return None


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

WEEKDAY_ALIASES = {
    "mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday",
    "weds": "Wednesday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
}


def parse_flexible_date(s):
    """Accept either MM-DD-YYYY or YYYY-MM-DD."""
    s = s.strip()
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"couldn't understand date '{s}' — use MM-DD-YYYY or YYYY-MM-DD")


def normalize_weekday(s):
    s_clean = s.strip().lower()
    if s_clean in WEEKDAY_ALIASES:
        return WEEKDAY_ALIASES[s_clean]
    for day in WEEKDAYS:
        if day.lower() == s_clean:
            return day
    return s.strip().title()  # left as-is; caller validates against WEEKDAYS


ADD_HELP_TEXT = (
    "To add a chore, use these fields (one 'key: value' per line):\n"
    "  name: <chore name>            (required)\n"
    "  assignee: <person>            (required)\n"
    "  type: recurring OR adhoc      (required)\n"
    "  due: MM-DD-YYYY               (required if type is adhoc)\n"
    "  frequency: daily / weekly / monthly / interval   (required if type is recurring)\n"
    "  days: Monday, Thursday        (required if frequency is weekly)\n"
    "  day_of_month: 1               (optional for monthly, default 1)\n"
    "  interval_days: 14             (required if frequency is interval)\n"
    "  start: MM-DD-YYYY             (optional for interval, default today)\n"
)


def build_chore_from_fields(data, fields):
    """Create a chore from a dict of string fields (as parsed from the CLI or an
    email). Appends it to `data` and returns the new chore dict, or raises
    ValueError with a human-readable message if something required is missing
    or invalid."""
    name = (fields.get("name") or "").strip()
    assignee = (fields.get("assignee") or "").strip()
    chore_type = (fields.get("type") or "").strip().lower()

    missing = [k for k, v in [("name", name), ("assignee", assignee), ("type", chore_type)] if not v]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if chore_type not in ("recurring", "adhoc"):
        raise ValueError("type must be 'recurring' or 'adhoc'")

    chore = {
        "id": data["next_id"],
        "name": name,
        "assignee": assignee,
        "type": chore_type,
        "last_completed": None,
    }

    if chore_type == "adhoc":
        due_raw = fields.get("due")
        if not due_raw:
            raise ValueError("adhoc chores need a 'due' date (MM-DD-YYYY)")
        chore["due_date"] = parse_flexible_date(due_raw).isoformat()
    else:
        freq = (fields.get("frequency") or "").strip().lower()
        if freq not in ("daily", "weekly", "monthly", "interval"):
            raise ValueError("frequency must be one of: daily, weekly, monthly, interval")
        chore["frequency"] = freq

        if freq == "weekly":
            days_raw = fields.get("days")
            if not days_raw:
                raise ValueError("weekly chores need a 'days' field, e.g. 'Monday, Thursday'")
            days = [normalize_weekday(d) for d in re.split(r"[,/]", days_raw) if d.strip()]
            bad = [d for d in days if d not in WEEKDAYS]
            if bad:
                raise ValueError(f"unrecognized weekday(s): {', '.join(bad)}")
            chore["days"] = days

        elif freq == "monthly":
            dom_raw = fields.get("day_of_month", "1")
            try:
                chore["day_of_month"] = int(dom_raw)
            except ValueError:
                raise ValueError("day_of_month must be a whole number (1-31)")

        elif freq == "interval":
            interval_raw = fields.get("interval_days")
            if not interval_raw:
                raise ValueError("interval chores need an 'interval_days' field, e.g. 14")
            try:
                chore["interval_days"] = int(interval_raw)
            except ValueError:
                raise ValueError("interval_days must be a whole number")
            start_raw = fields.get("start")
            chore["start_date"] = parse_flexible_date(start_raw).isoformat() if start_raw else today().isoformat()

    data["chores"].append(chore)
    data["next_id"] += 1
    return chore


def cmd_add(args):
    data = load_data()
    fields = {
        "name": args.name,
        "assignee": args.assignee,
        "type": args.type,
        "due": args.due,
        "frequency": args.frequency,
        "days": args.days,
        "day_of_month": str(args.day_of_month) if args.day_of_month is not None else None,
        "interval_days": str(args.interval_days) if args.interval_days is not None else None,
        "start": args.start,
    }
    try:
        chore = build_chore_from_fields(data, fields)
    except ValueError as e:
        print(f"Couldn't add chore: {e}")
        sys.exit(1)

    save_data(data)
    pos = display_position(data, chore)
    print(f"Added chore #{pos}: {chore['name']} (assigned to {chore['assignee']})")


def format_date_mdy(d):
    return d.strftime("%m-%d-%Y")


def describe_recurrence(chore):
    """Human-readable description of how often a chore recurs, e.g.
    'every day', 'every Monday', 'every 2 weeks', 'every month on day 1'."""
    if chore["type"] == "adhoc":
        return "one-time"

    freq = chore["frequency"]

    if freq == "daily":
        return "every day"

    if freq == "weekly":
        days = chore.get("days", [])
        if not days:
            return "weekly"
        if len(days) == 1:
            return f"every {days[0]}"
        if len(days) == 2:
            return f"every {days[0]} and {days[1]}"
        return "every " + ", ".join(days[:-1]) + f", and {days[-1]}"

    if freq == "monthly":
        dom = chore.get("day_of_month", 1)
        return f"every month on day {dom}"

    if freq == "interval":
        interval = chore.get("interval_days", 7)
        if interval == 7:
            return "every week"
        if interval % 7 == 0:
            return f"every {interval // 7} weeks"
        if interval == 1:
            return "every day"
        return f"every {interval} days"

    return freq


def format_chore_line(position, c, on_day):
    status = get_status(c, on_day)
    if c["type"] == "adhoc":
        detail = f"due {format_date_mdy(parse_date(c['due_date']))}"
    else:
        detail = describe_recurrence(c)
    next_due = get_next_due(c, on_day)
    next_str = f"next due: {format_date_mdy(next_due)}" if next_due else "no further occurrences (one-time chore)"
    return f"#{position:<3} [{status:<9}] {c['name']:<25} -> {c['assignee']:<10} ({detail}) | {next_str}"


def build_display_groups(data, on_day=None):
    """Chores grouped and ordered the way they're always shown and referenced
    by position: (overdue, due, done). Concatenating these three lists in
    order gives the full display ordering -- position N (1-based) in that
    concatenation is what every position-based lookup (resolve_position,
    build_status_report's numbering, remove_chores) means by "position N".
    This is the single source of truth every one of them derives from, so
    they can never disagree about what position N currently refers to."""
    on_day = on_day or today()
    chores = data["chores"]
    overdue = [c for c in chores if get_status(c, on_day) == "OVERDUE"]
    due = [c for c in chores if get_status(c, on_day) == "DUE"]
    done = [c for c in chores if get_status(c, on_day) == "DONE"]
    return overdue, due, done


def resolve_position(data, position, on_day=None):
    """Resolve a 1-based display position (as shown by build_status_report,
    overdue first, then due, then done) to the chore it CURRENTLY refers to.

    IMPORTANT: positions are recomputed fresh from current status every
    time this is called -- they are not stable ids. If the list has
    changed since a position number was last shown (a chore added or
    removed, or ANY chore's status flipping, e.g. simply because a day
    passed), the same number can point to a different chore than it did
    before. This is the deliberate trade-off of always-consecutive display
    numbering (1, 2, 3... starting at OVERDUE) instead of stable per-chore
    ids: simpler, tidier listings, at the cost of a 'done <n>'/'remove <n>'
    reply to an OLD email only being reliable if nothing about the list
    has changed since that email was generated. Returns None if the
    position doesn't currently exist."""
    overdue, due, done = build_display_groups(data, on_day)
    order = overdue + due + done
    idx = position - 1
    if idx < 0 or idx >= len(order):
        return None
    return order[idx]


def display_position(data, chore, on_day=None):
    """The given chore's current 1-based display position, for confirmation
    messages right after creating a chore (where there's no user-supplied
    position to just echo back)."""
    overdue, due, done = build_display_groups(data, on_day)
    for i, c in enumerate(overdue + due + done, start=1):
        if c is chore:
            return i
    return None


def mark_chore_done(data, position):
    """Mark the chore currently AT the given display position (see
    resolve_position()) as done today. Returns the chore dict, or None if
    that position doesn't currently exist."""
    chore = resolve_position(data, position)
    if chore is None:
        return None
    chore["last_completed"] = today().isoformat()
    return chore


def build_status_report(data, assignee=None):
    """Plain-text status listing, used for both the terminal `list` command
    and the reply sent when someone emails 'list' or 'status'. Grouped the
    same way as the reminder email (overdue first, then due, then a
    separate done section), and numbered the same way too: a single
    consecutive count (1, 2, 3...) starting at OVERDUE and running through
    DUE then DONE. That numbering is always computed from the FULL,
    unfiltered chore list (see build_display_groups) even when `assignee`
    filters what's actually printed -- so a number shown in a filtered
    ('list Alice') view still resolves correctly in a 'done <n>' reply.

    If `assignee` is given, only that person's chores are displayed
    (case-insensitive, exact match on the assignee field) -- but note the
    numbers next to them are their GLOBAL position, not a fresh 1..N count
    of just their chores."""
    chores = data["chores"]
    if assignee:
        if not any(c["assignee"].strip().lower() == assignee.strip().lower() for c in chores):
            return f"No chores found for '{assignee}'."

    if not chores:
        return "No chores yet."

    on_day = today()
    overdue_all, due_all, done_all = build_display_groups(data, on_day)
    positions = {id(c): i for i, c in enumerate(overdue_all + due_all + done_all, start=1)}

    def matches(c):
        return not assignee or c["assignee"].strip().lower() == assignee.strip().lower()

    overdue = [c for c in overdue_all if matches(c)]
    due = [c for c in due_all if matches(c)]
    done = [c for c in done_all if matches(c)]

    header = f"Current chore status for {assignee}:" if assignee else "Current chore status:"
    lines = [header, ""]

    has_attention_section = bool(overdue or due)
    if has_attention_section:
        lines.append("NEEDS ATTENTION:")
        for c in overdue:
            lines.append(format_chore_line(positions[id(c)], c, on_day))
        for c in due:
            lines.append(format_chore_line(positions[id(c)], c, on_day))

    if done:
        if has_attention_section:
            lines.append("")
            lines.append("")
        lines.append("DONE:")
        for c in done:
            lines.append(format_chore_line(positions[id(c)], c, on_day))

    lines.append("")
    lines.append("Reply 'done <number>' (e.g. 'done 3') to mark a chore complete, or 'list' to see this again.")
    return "\n".join(lines)


def cmd_list(args):
    data = load_data()
    print(build_status_report(data, assignee=args.assignee))


def cmd_done(args):
    data = load_data()
    c = mark_chore_done(data, args.position)
    if c:
        save_data(data)
        print(f"Marked '{c['name']}' (was #{args.position}) as done today.")
    else:
        print(f"No chore at position {args.position} -- run 'list' to see current numbers.")


def remove_chores(data, positions):
    """Remove multiple chores by their current display position (see
    resolve_position()), in one atomic step: resolve every position to a
    chore from a SINGLE freshly-computed ordering *before* removing or
    renumbering anything, so removing several positions in one command
    can't be thrown off by positions shifting mid-operation. Returns
    (removed, not_found) where `removed` is a list of (position, chore)
    pairs -- the position each chore was removed at, paired with its data
    -- and `not_found` is a list of positions that didn't resolve to
    anything."""
    overdue, due, done = build_display_groups(data)
    order = overdue + due + done

    removed, not_found, seen = [], [], set()
    for position in positions:
        if position in seen:
            continue
        seen.add(position)
        idx = position - 1
        match = order[idx] if 0 <= idx < len(order) else None
        if match is not None:
            removed.append((position, match))
        else:
            not_found.append(position)

    for _, c in removed:
        data["chores"].remove(c)

    if removed:
        for new_id, c in enumerate(data["chores"], start=1):
            c["id"] = new_id
        data["next_id"] = len(data["chores"]) + 1

    return removed, not_found


def remove_chore(data, position):
    """Remove a single chore by its current display position. Returns the
    removed chore dict, or None if no chore was at that position."""
    removed, _ = remove_chores(data, [position])
    return removed[0][1] if removed else None


def cmd_remove(args):
    data = load_data()
    c = remove_chore(data, args.position)
    if c:
        save_data(data)
        print(f"Removed '{c['name']}' (was #{args.position})")
    else:
        print(f"No chore at position {args.position} -- run 'list' to see current numbers.")


def build_email_body(due_chores, overdue_chores, done_chores):
    lines = []
    lines.append(f"Chore reminder for {today().strftime('%A, %B %d, %Y')}")
    lines.append("")

    n = 1
    has_attention_section = bool(overdue_chores or due_chores)
    if has_attention_section:
        lines.append("NEEDS ATTENTION:")
        for c in overdue_chores:
            start = get_scheduled_start(c, today())
            lines.append(f"  {n}. [OVERDUE] {c['name']} (assigned to {c['assignee']}, due since {format_date_mdy(start)})")
            n += 1
        for c in due_chores:
            lines.append(f"  {n}. [DUE] {c['name']} (assigned to {c['assignee']})")
            n += 1

    if done_chores:
        if has_attention_section:
            lines.append("")
            lines.append("")
        lines.append("DONE:")
        for c in done_chores:
            lines.append(f"  {n}. {c['name']} (assigned to {c['assignee']})")
            n += 1

    if not due_chores and not overdue_chores and not done_chores:
        lines.append("Nothing due. Nice work!")

    lines.append("")
    lines.append("Mark chores done with: python3 chores.py done <number>")
    return "\n".join(lines)


def send_email(subject, body, config, to_list=None):
    """Send a plain (non-reply) automated email. Defaults to broadcasting
    to every address in config["recipients"] -- pass an explicit `to_list`
    to send to specific address(es) instead (see cmd_remind's per-assignee
    routing below)."""
    to_list = to_list if to_list is not None else config["recipients"]
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config["sender_email"]
    msg["To"] = ", ".join(to_list)
    msg[BOT_HEADER] = "yes"

    with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
        server.starttls()
        server.login(config["sender_email"], config["sender_password"])
        server.sendmail(config["sender_email"], to_list, msg.as_string())


def get_assignee_email(config, assignee):
    """Look up an assignee's email address from config["assignee_emails"]
    (a {name: email} dict, matched case-insensitively against the
    'assignee' field). Returns None if there's no entry -- callers decide
    how to handle that (see cmd_remind)."""
    directory = config.get("assignee_emails") or {}
    lookup = {name.strip().lower(): addr for name, addr in directory.items()}
    return lookup.get(assignee.strip().lower())


def cmd_remind(args):
    data = load_data()
    config = load_config()

    due = [c for c in data["chores"] if get_status(c, today()) == "DUE"]
    overdue = [c for c in data["chores"] if get_status(c, today()) == "OVERDUE"]
    done = [c for c in data["chores"] if get_status(c, today()) == "DONE"]

    if not due and not overdue:
        print("Nothing needs attention today — no reminder email sent.")
        return

    if not config.get("assignee_emails"):
        # No per-assignee directory configured -- unchanged legacy
        # behavior: one broadcast email to everyone in config["recipients"].
        body = build_email_body(due, overdue, done)
        subject = tag_ai_generated(f"Chore reminder - {len(due) + len(overdue)} item(s) need attention")
        send_email(subject, body, config)
        print("Reminder email sent.")
        print(body)
        return

    # assignee_emails IS configured -- send each assignee their own
    # reminder, containing only their own chores, at their own address.
    # Assignees with no due/overdue items today aren't emailed at all
    # (nothing needs their attention); their DONE items still get included
    # for context in whichever email they do get.
    assignees_needing_attention = sorted({c["assignee"] for c in due + overdue})
    unmapped = []

    for assignee in assignees_needing_attention:
        a_due = [c for c in due if c["assignee"] == assignee]
        a_overdue = [c for c in overdue if c["assignee"] == assignee]
        a_done = [c for c in done if c["assignee"] == assignee]
        body = build_email_body(a_due, a_overdue, a_done)
        subject = tag_ai_generated(f"Chore reminder - {len(a_due) + len(a_overdue)} item(s) need attention")

        to_addr = get_assignee_email(config, assignee)
        if to_addr:
            send_email(subject, body, config, to_list=[to_addr])
            print(f"Reminder sent to {assignee} <{to_addr}>.")
        else:
            unmapped.append((assignee, body))

    if unmapped:
        # Never silently drop a reminder just because someone's missing
        # from the directory -- bundle them into one email to the account
        # owner instead, with a clear nudge to fix the config.
        names = ", ".join(a for a, _ in unmapped)
        print(f"No email on file for: {names}. Add them to config.json's "
              f"'assignee_emails' -- routing their reminder(s) to "
              f"{config['sender_email']} for now.")
        combined_body = "\n\n---\n\n".join(f"[{a}]\n{b}" for a, b in unmapped)
        subject = tag_ai_generated(f"Chore reminder - no email on file for: {names}")
        send_email(subject, combined_body, config, to_list=[config["sender_email"]])

    print("Reminder email(s) sent.")


# --------------------------------------------------------------------------
# Incoming email: replying to "list"/"status" and "done <number>"
# --------------------------------------------------------------------------

def send_reply_email(to_addr, subject, body, config, in_reply_to=None, references=None):
    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject
    subject = tag_ai_generated(subject)
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config["sender_email"]
    msg["To"] = to_addr
    msg[BOT_HEADER] = "yes"

    # These are what actually make an email client (Gmail, Outlook, etc.)
    # group this as part of the same conversation, rather than just
    # relying on the subject line looking similar.
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
        server.starttls()
        server.login(config["sender_email"], config["sender_password"])
        server.sendmail(config["sender_email"], [to_addr], msg.as_string())


def get_email_body_text(msg):
    """Pull the plain-text body out of an email.message.Message, multipart or not."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True) or b""
                return payload.decode(charset, errors="replace")
        return ""
    charset = msg.get_content_charset() or "utf-8"
    payload = msg.get_payload(decode=True) or b""
    return payload.decode(charset, errors="replace")


def strip_quoted_reply(body):
    """Keep only the part of the email the person actually typed, dropping
    quoted history below it (lines starting with '>' or an 'On ... wrote:' marker)."""
    kept = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            break
        if re.match(r"^On .+ wrote:$", stripped):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def parse_key_value_lines(text):
    """Parse lines like 'name: Take out trash' into a {key: value} dict.
    Keys are lowercased with spaces turned to underscores (e.g. 'day of month' -> 'day_of_month')."""
    fields = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z_ ]+?)\s*[:=]\s*(.+?)\s*$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            value = m.group(2).strip()
            fields[key] = value
    return fields


def parse_command(body):
    """Command detection: 'add' (with key:value fields), 'remove <numbers>',
    'done <numbers>', or 'list'/'status' (optionally followed by a person's
    name to filter, e.g. 'list Alice' or 'status for Bob'). Returns (command, arg)."""
    raw = strip_quoted_reply(body)
    if not raw:
        return None, None
    text = raw.lower()

    if re.match(r"^\s*add\b", text):
        fields = parse_key_value_lines(raw)
        if fields:
            # At least one recognizable 'key: value' line -- this is someone
            # attempting the structured syntax (even if incompletely), so
            # handle it here for free/instantly rather than involving the LLM.
            return "add", fields
        # Starts with "add" but has NO key:value fields at all -- almost
        # certainly a natural-language request ("add chore X for Bob, due
        # every Thursday") rather than an attempt at the structured format.
        # Don't claim it here: returning (None, None) lets this fall through
        # to the tool-using LLM agent (if one is configured), which can
        # actually interpret the request and ask a clarifying question if
        # something's genuinely missing, instead of everyone who phrases an
        # add request in plain English instantly getting the raw field list.
        return None, None

    if "remove" in text or "delete" in text:
        ids = [int(n) for n in re.findall(r"\d+", text)]
        if ids:
            return "remove", ids

    if "done" in text:
        ids = [int(n) for n in re.findall(r"\d+", text)]
        if ids:
            return "done", ids

    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    name_match = re.match(r"^(?:list|status)\b\s*(?:for|of)?\s*(.*)$", first_line, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        return "list", (name or None)

    if "list" in text or "status" in text:
        return "list", None

    return None, None


def try_regex_command(body, data):
    """Try to handle one incoming email body using ONLY the plain-text regex
    commands (add/remove/done/list) -- no LLM involved at all. Returns
    (reply_text, data_changed) if a command was recognized, or None if
    nothing matched.

    This is deliberately split out of handle_incoming_email() so the same
    detect-and-execute logic can be reused in two places: here, as the
    degraded-mode (no orchestrator configured) path, and in
    orchestrator.py's chores_node, as the fast/free path a router-first
    flow still gets to use once the router has already decided a message
    belongs to the chores domain. Keeping one copy of this logic means the
    two call sites can't drift apart."""
    command, arg = parse_command(body)

    if command == "add":
        try:
            chore = build_chore_from_fields(data, arg)
        except ValueError as e:
            return f"Couldn't add chore: {e}\n\n{ADD_HELP_TEXT}", False
        pos = display_position(data, chore)
        lines = [f"Added chore #{pos}: {chore['name']} (assigned to {chore['assignee']})", ""]
        lines.append(build_status_report(data))
        return "\n".join(lines), True

    if command == "remove":
        # NOTE: `arg` here holds display positions (see resolve_position()'s
        # docstring on why a position isn't a stable id) -- they're resolved
        # against the list as it stands right now, which is correct for a
        # reply to a list you just saw, but can point at the wrong chore if
        # you're replying to an older email after the list has changed.
        removed, not_found = remove_chores(data, arg)

        lines = []
        if removed:
            lines.append("Removed: " + ", ".join(f"#{pos} {c['name']}" for pos, c in removed))
        if not_found:
            lines.append("Couldn't find chore number(s): " + ", ".join(str(i) for i in not_found))
        lines.append("")
        lines.append(build_status_report(data))
        return "\n".join(lines), bool(removed)

    if command == "done":
        # Same position-not-stable-id caveat as "remove" above.
        marked, not_found = [], []
        for position in arg:
            c = mark_chore_done(data, position)
            if c:
                marked.append((position, c))
            else:
                not_found.append(position)

        lines = []
        if marked:
            lines.append("Marked done: " + ", ".join(f"#{pos} {c['name']}" for pos, c in marked))
        if not_found:
            lines.append("Couldn't find chore number(s): " + ", ".join(str(i) for i in not_found))
        lines.append("")
        lines.append(build_status_report(data))
        return "\n".join(lines), bool(marked)

    if command == "list":
        return build_status_report(data, assignee=arg), False

    return None


def build_help_text():
    return (
        "Sorry, I didn't catch a command in that reply.\n\n"
        "Reply 'list' or 'status' to see current chore status (add a name, "
        "e.g. 'list Alice', to see just one person's chores), "
        "'done <number>' (e.g. 'done 3', using the number from the list) "
        "to mark a chore complete, 'remove <number>' to delete a chore, or "
        "'add' followed by chore fields to create one — see below.\n\n" + ADD_HELP_TEXT
    )


def handle_incoming_email(body, data, sender=None):
    """Interpret one incoming email's body and return (reply_text, data_changed).

    This is the ORIGINAL (pre-goals) regex-first-then-LLM-then-help flow,
    kept exactly as it always behaved -- same control flow, same order of
    checks. It's still used directly by route_message() below for the "no
    orchestrator configured at all" case. It's no longer the primary entry
    point once an orchestrator IS configured; see route_message()."""
    result = try_regex_command(body, data)
    if result is not None:
        return result

    # No regex command matched — hand off to the LLM orchestrator, if one is
    # configured. It manages its own persistence (its tools call load_data/
    # save_data directly), so we always report changed=False here; the
    # caller (check_email_once) reloads data fresh after this returns so it
    # never overwrites what the orchestrator just saved.
    llm_reply = try_llm_orchestrator(body, sender)
    if llm_reply is not None:
        return llm_reply, False

    return build_help_text(), False


# --------------------------------------------------------------------------
# Confirm-before-delete
# --------------------------------------------------------------------------
#
# Deleting a chore by exact position (the plain-text "remove <number>"
# command handled in try_regex_command) stays instant/unchanged -- someone
# typing an exact number is already looking at a specific list and knows
# exactly what they're removing.
#
# Natural-language deletion requests handled by the LLM agent (e.g. "delete
# the trash chore for Bob") are different: the LLM has to interpret what the
# person means, and interpretation can be wrong. So that path never deletes
# immediately -- it stages a PENDING deletion (via propose_deletion() below,
# called from orchestrator.py's propose_delete_chore tool) and waits for the
# person to explicitly confirm in a follow-up reply before anything is
# actually removed. See handle_pending_deletion_reply(), which route_message()
# checks first, before any other routing, so a stray "yes"/"no" reply can't
# get swallowed by the router or misread as some other command.

CONFIRM_WORDS = {
    "yes", "y", "yes please", "confirm", "confirmed", "go ahead", "go for it",
    "do it", "please do", "correct", "sure", "yep", "yeah", "ok", "okay",
}
CANCEL_WORDS = {
    "no", "n", "cancel", "nevermind", "never mind", "stop", "dont", "do not",
    "abort", "wait",
}


def _matches_word_list(text, words):
    """True if `text` (after stripping punctuation) IS one of `words`, or
    STARTS WITH one of them followed by more words (e.g. "yes delete them"
    matches "yes", "never mind, leave it" matches "never mind"). Comparing
    on the leading phrase rather than requiring the whole message to be
    exactly one of these words keeps this forgiving of how people actually
    reply to a confirmation email."""
    normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not normalized:
        return False
    for w in words:
        w_norm = re.sub(r"[^\w\s]", "", w)
        if normalized == w_norm or normalized.startswith(w_norm + " "):
            return True
    return False


def looks_like_confirmation(text):
    return _matches_word_list(text, CONFIRM_WORDS)


def looks_like_cancellation(text):
    return _matches_word_list(text, CANCEL_WORDS)


def load_pending_deletions():
    if not os.path.exists(PENDING_DELETIONS_FILE):
        return {}
    with open(PENDING_DELETIONS_FILE, "r") as f:
        return json.load(f)


def save_pending_deletions(state):
    with open(PENDING_DELETIONS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def set_pending_deletion(sender, items):
    """Stage a pending deletion for `sender` (their lowercased email address
    is the key), replacing any earlier pending deletion they had. `items`
    is a list of {"name", "assignee"} dicts identifying the chores to
    delete -- see resolve_pending_items() for why this stores content
    (name + assignee) rather than a position or the internal id: both of
    those can point somewhere else by the time a confirmation reply comes
    back (positions are recomputed fresh every call per resolve_position()'s
    docstring, and internal ids get renumbered whenever ANY chore is
    removed, see remove_chores())."""
    state = load_pending_deletions()
    state[sender.strip().lower()] = {
        "items": items,
        "proposed_at": datetime.now().isoformat(),
    }
    save_pending_deletions(state)


def get_pending_deletion(sender):
    """Returns the pending-deletion entry for `sender`, or None if there
    isn't one or it's expired (older than PENDING_DELETION_TTL_HOURS -- a
    stale confirmation shouldn't silently execute against a list that's
    since moved on). An expired entry is cleared as a side effect."""
    state = load_pending_deletions()
    key = sender.strip().lower()
    entry = state.get(key)
    if entry is None:
        return None
    proposed_at = datetime.fromisoformat(entry["proposed_at"])
    age_hours = (datetime.now() - proposed_at).total_seconds() / 3600
    if age_hours > PENDING_DELETION_TTL_HOURS:
        clear_pending_deletion(sender)
        return None
    return entry


def clear_pending_deletion(sender):
    state = load_pending_deletions()
    state.pop(sender.strip().lower(), None)
    save_pending_deletions(state)


def resolve_pending_items(data, items):
    """Re-resolve each {"name", "assignee"} item against the CURRENT chore
    list. Deliberately matches on name+assignee content rather than
    position or internal id (see set_pending_deletion()'s docstring for
    why). Returns (resolved, unresolved, ambiguous):
      resolved   -- list of chore dicts, one confident match per item
      unresolved -- items with zero matches (e.g. already deleted)
      ambiguous  -- items with 2+ matches -- skipped rather than guessed;
                    the person can use 'remove <number>' for these instead
    """
    resolved, unresolved, ambiguous = [], [], []
    for item in items:
        name = (item.get("name") or "").strip().lower()
        assignee = (item.get("assignee") or "").strip().lower()
        matches = [c for c in data["chores"]
                   if c["name"].strip().lower() == name
                   and c["assignee"].strip().lower() == assignee]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) == 0:
            unresolved.append(item)
        else:
            ambiguous.append(item)
    return resolved, unresolved, ambiguous


def propose_deletion(data, sender, items):
    """Stage a pending deletion for `sender` and build a confirmation
    preview. `items` is a list of {"name", "assignee"} dicts identifying
    chores to delete -- looked up by content, NOT position numbers (see
    set_pending_deletion()'s docstring for why). Resolves each item against
    the chore list right now so the preview reflects reality, stores the
    same name/assignee identifiers in pending_deletions.json, and returns
    (preview_text, matched_count). Nothing is deleted here -- deletion only
    happens if/when the person confirms (see handle_pending_deletion_reply).
    If nothing resolves, no pending deletion is stored and matched_count
    is 0."""
    resolved, unresolved, ambiguous = resolve_pending_items(data, items)

    if not resolved:
        lines = ["I couldn't find any chores matching what you asked to delete."]
        if unresolved:
            lines.append("Not found: " + ", ".join(f"{i['name']} ({i['assignee']})" for i in unresolved))
        if ambiguous:
            lines.append("Matched more than one chore (ambiguous): " +
                          ", ".join(f"{i['name']} ({i['assignee']})" for i in ambiguous))
        lines.append("")
        lines.append(build_status_report(data))
        return "\n".join(lines), 0

    on_day = today()
    lines = ["I'm about to delete the following chore(s):", ""]
    for c in resolved:
        pos = display_position(data, c, on_day)
        lines.append(f"  #{pos} {c['name']} -> {c['assignee']}")
    if unresolved:
        lines.append("")
        lines.append("Couldn't find (skipping): " + ", ".join(f"{i['name']} ({i['assignee']})" for i in unresolved))
    if ambiguous:
        lines.append("")
        lines.append("Ambiguous, matched more than one chore with the same name/assignee "
                      "(skipping -- use 'remove <number>' for these instead): " +
                      ", ".join(f"{i['name']} ({i['assignee']})" for i in ambiguous))
    lines.append("")
    lines.append("Reply 'yes' to confirm and delete these, or 'no' to cancel.")

    items_to_store = [{"name": c["name"], "assignee": c["assignee"]} for c in resolved]
    set_pending_deletion(sender, items_to_store)
    return "\n".join(lines), len(resolved)


def handle_pending_deletion_reply(body, data, sender):
    """If `sender` has a pending deletion awaiting confirmation, interpret
    this reply as answering it (confirm / cancel / unclear) and return
    (reply_text, data_changed). Returns None if there's no pending deletion
    for this sender -- callers should fall through to normal routing in
    that case. Checked FIRST in route_message(), before any regex/LLM
    routing, so a stray 'yes' can't be misrouted as some other command."""
    pending = get_pending_deletion(sender)
    if pending is None:
        return None

    text = strip_quoted_reply(body)

    if looks_like_cancellation(text):
        clear_pending_deletion(sender)
        return ("Okay, I won't delete anything. Nothing was changed.\n\n" +
                build_status_report(data)), False

    if not looks_like_confirmation(text):
        items = pending["items"]
        names = ", ".join(f"{i['name']} ({i['assignee']})" for i in items)
        return (f"Still waiting on confirmation to delete: {names}.\n"
                f"Reply 'yes' to confirm, or 'no' to cancel."), False

    resolved, unresolved, ambiguous = resolve_pending_items(data, pending["items"])
    clear_pending_deletion(sender)

    if not resolved:
        lines = ["Nothing was deleted -- I couldn't re-find any of the chores I'd proposed "
                  "(the list may have changed since). Here's the current status:", ""]
        lines.append(build_status_report(data))
        return "\n".join(lines), False

    for c in resolved:
        data["chores"].remove(c)
    for new_id, c in enumerate(data["chores"], start=1):
        c["id"] = new_id
    data["next_id"] = len(data["chores"]) + 1

    lines = ["Deleted: " + ", ".join(f"{c['name']} ({c['assignee']})" for c in resolved)]
    if unresolved:
        lines.append("Couldn't find (skipped): " + ", ".join(f"{i['name']} ({i['assignee']})" for i in unresolved))
    if ambiguous:
        lines.append("Ambiguous, skipped: " + ", ".join(f"{i['name']} ({i['assignee']})" for i in ambiguous))
    lines.append("")
    lines.append(build_status_report(data))
    return "\n".join(lines), True


def route_message(body, data, sender, config):
    """Router-first entry point used by check_email_once() whenever an LLM
    orchestrator is configured.

    Why this exists (separately from handle_incoming_email): once there's
    more than one LLM-backed domain (chores, goals, ...), a domain-specific
    regex fast path can no longer be allowed to run BEFORE the router gets
    a chance to classify the message -- otherwise a goals-related email
    that happens to contain a word like "done" or "list" gets silently
    claimed as a chore command and the router never even sees it. (This is
    exactly what happened during testing before this fix.)

    So: when an orchestrator is available, this hands the raw body straight
    to it -- orchestrator.handle_message() runs the router FIRST, and each
    specialist node is responsible for its own fast path internally, scoped
    to messages the router already decided belong to that domain (see
    chores_node in orchestrator.py, which calls try_regex_command() itself
    before falling back to its tool-using agent). Only when the
    orchestrator is unavailable/fails for this message do we fall back to
    plain regex + help text -- and we do that directly here (not by
    re-calling handle_incoming_email, which would retry the LLM call a
    second time and double up on cost/latency for a call we already just
    watched fail).

    With no API key configured at all, this is a pure pass-through to
    handle_incoming_email() -- identical to pre-goals behavior.

    One check runs before ANY of that, regardless of API key: if `sender`
    has a pending, unconfirmed deletion staged (see propose_deletion() /
    handle_pending_deletion_reply()), this reply is treated as the answer
    to that confirmation, full stop -- it never reaches the router or the
    regex fast path, so a plain "yes" can't be misread as some unrelated
    command."""
    pending_reply = handle_pending_deletion_reply(body, data, sender)
    if pending_reply is not None:
        return pending_reply

    api_key = config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return handle_incoming_email(body, data, sender)

    llm_reply = try_llm_orchestrator(body, sender)
    if llm_reply is not None:
        return llm_reply, False

    # Orchestrator is configured but unavailable or failed for this message
    # (missing deps, API error, etc.) -- try_llm_orchestrator() already
    # printed why. Fall back to the deterministic regex path, then help
    # text, without retrying the LLM call.
    result = try_regex_command(body, data)
    if result is not None:
        return result
    return build_help_text(), False


def try_llm_orchestrator(body, sender):
    """Attempt to handle a message via the LangGraph LLM orchestrator (see
    orchestrator.py), if one is configured. Returns the reply text, or None
    if the orchestrator isn't available/configured — callers should fall
    back to the plain help text in that case. Never raises: any failure
    (missing dependency, missing API key, API error) degrades gracefully."""
    config = load_config()
    api_key = config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import orchestrator
    except ImportError as e:
        print(f"LLM orchestrator unavailable ({e}). "
              f"Install dependencies with: pip install -r requirements.txt")
        return None

    try:
        return orchestrator.handle_message(body, sender, api_key)
    except Exception as e:
        print(f"LLM orchestrator error, falling back to plain help text: {e}")
        return None


def is_chores_only_subject(subject):
    """True only if the subject is exactly the word 'chores' (nothing before/after)."""
    return subject.strip().lower() == "chores"


def is_bot_thread_reply(subject):
    """True if this is a reply ('Re:') within a thread the bot itself started —
    i.e. the subject still carries our [AI generated] tag, inherited from
    whatever automated email kicked off the thread (a reminder, or an earlier
    reply to a 'chores'/'add'/'done'/etc. email). This intentionally does NOT
    match arbitrary 'Re:' emails on unrelated subjects — only ones descended
    from something we sent."""
    s = subject.strip()
    if not re.match(r"^(re\s*:\s*)+", s, re.IGNORECASE):
        return False  # must actually be a reply
    return AI_TAG.lower() in s.lower()


def check_email_once():
    """Check the inbox once, process any accepted emails, and reply. Returns
    the number of emails processed (0 if nothing new or nothing accepted).

    Uses IMAP UIDs (not the \\Seen flag) to track what's already been looked
    at, since \\Seen can get set by a webmail client (e.g. opening a thread
    to reply to it) before this script ever sees the message. On the very
    first run there's no baseline yet, so we record the mailbox's current
    UIDNEXT and process nothing — from then on, only messages that arrive
    *after* that point are ever considered, so a mailbox with years of
    history is never scanned."""
    config = load_config()

    # Anyone in config["recipients"] can command the bot, same as always --
    # PLUS anyone with an address on file in config["assignee_emails"], so a
    # household member gets reply access to their own reminders without
    # needing to be listed in both places.
    allowed_senders = {addr.lower() for addr in config.get("recipients", [])}
    allowed_senders |= {addr.lower() for addr in (config.get("assignee_emails") or {}).values()}
    imap_host = config.get("imap_host", config["smtp_host"].replace("smtp", "imap", 1))
    imap_port = config.get("imap_port", 993)

    imap = imaplib.IMAP4_SSL(imap_host, imap_port)
    try:
        imap.login(config["sender_email"], config["sender_password"])
        imap.select("INBOX")

        state = load_email_state()

        if state.get("last_uid") is None:
            status, status_data = imap.status("INBOX", "(UIDNEXT)")
            if status != "OK":
                print("Could not read inbox UIDNEXT to initialize tracking.")
                return 0
            match = re.search(r"UIDNEXT (\d+)", status_data[0].decode())
            uidnext = int(match.group(1)) if match else 1
            save_email_state({"last_uid": uidnext - 1})
            print(f"Initialized email tracking at UID {uidnext - 1}. "
                  f"Only messages arriving from now on will be checked.")
            return 0

        last_uid = state["last_uid"]
        status, msg_nums = imap.uid("search", None, f"UID {last_uid + 1}:*")
        if status != "OK":
            print("Could not search inbox for new messages.")
            return 0

        # IMAP quirk: "UID X:*" with X past the highest UID in the mailbox
        # can still return the single newest message. Filter those out —
        # we only want UIDs strictly greater than what we've already seen.
        uids = [int(u) for u in msg_nums[0].split() if int(u) > last_uid]
        if not uids:
            return 0

        processed = 0
        max_uid_seen = last_uid

        for uid in uids:
            max_uid_seen = max(max_uid_seen, uid)

            status, msg_data = imap.uid("fetch", str(uid), "(BODY.PEEK[])")
            if status != "OK" or not msg_data or not msg_data[0]:
                print(f"UID {uid}: could not fetch message, skipping.")
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            from_addr = email.utils.parseaddr(msg.get("From", ""))[1].lower()
            subject = msg.get("Subject", "(no subject)")

            if allowed_senders and from_addr not in allowed_senders:
                print(f"UID {uid}: ignoring — sender {from_addr!r} not in recipients list. Leaving unread.")
                continue

            if msg.get(BOT_HEADER):
                print(f"UID {uid}: ignoring — has {BOT_HEADER} header (sent by this bot). "
                      f"Leaving unread. Subject: {subject!r}")
                continue

            if not (is_chores_only_subject(subject) or is_bot_thread_reply(subject)):
                print(f"UID {uid}: ignoring — subject doesn't match an accepted pattern. "
                      f"Leaving unread. Subject: {subject!r}")
                continue

            # Reloaded fresh for every message rather than once for the whole
            # batch: the LLM orchestrator's tools persist their own changes
            # directly (see try_llm_orchestrator), so a stale in-memory copy
            # from an earlier iteration could otherwise overwrite them.
            body = get_email_body_text(msg)
            data = load_data()
            # route_message() runs the LLM router FIRST when an orchestrator
            # is configured (so chores/goals/etc. can't cross-contaminate on
            # keyword overlap), and falls back to the original regex-first
            # behavior when it isn't. See route_message()'s docstring.
            reply_text, changed = route_message(body, data, sender=from_addr, config=config)
            if changed:
                save_data(data)
            processed += 1

            original_message_id = msg.get("Message-ID")
            references = (msg.get("References", "").strip() + " " + (original_message_id or "")).strip()
            send_reply_email(from_addr, subject, reply_text, config,
                              in_reply_to=original_message_id, references=references or None)

            # Only now — having actually taken action on it — mark it read.
            imap.uid("store", str(uid), "+FLAGS", "(\\Seen)")
            print(f"UID {uid}: replied to {from_addr} and marked read (subject: {subject!r})")

        save_email_state({"last_uid": max_uid_seen})

        return processed
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def cmd_check_email(args):
    n = check_email_once()
    print("No new emails to process." if n == 0 else f"Processed {n} email(s).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Household chore tracker with email reminders")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new chore")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--assignee", required=True)
    p_add.add_argument("--type", choices=["recurring", "adhoc"], required=True)
    p_add.add_argument("--due", help="YYYY-MM-DD, required for adhoc")
    p_add.add_argument("--frequency", choices=["daily", "weekly", "monthly", "interval"])
    p_add.add_argument("--days", help="Comma-separated weekday names, e.g. Monday,Thursday")
    p_add.add_argument("--day-of-month", dest="day_of_month", type=int)
    p_add.add_argument("--interval-days", dest="interval_days", type=int)
    p_add.add_argument("--start", help="YYYY-MM-DD start date for interval chores")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List all chores and their status")
    p_list.add_argument("--assignee", help="Only show chores for this person")
    p_list.set_defaults(func=cmd_list)

    p_done = sub.add_parser("done", help="Mark a chore as completed today")
    p_done.add_argument("position", type=int, help="Position number from 'list' (overdue, then due, then done)")
    p_done.set_defaults(func=cmd_done)

    p_remove = sub.add_parser("remove", help="Delete a chore")
    p_remove.add_argument("position", type=int, help="Position number from 'list' (overdue, then due, then done)")
    p_remove.set_defaults(func=cmd_remove)

    p_remind = sub.add_parser("remind", help="Email a daily chore status (overdue/due/done)")
    p_remind.set_defaults(func=cmd_remind)

    p_check_email = sub.add_parser("check-email", help="Check inbox once for replies ('list'/'done <number>') and respond")
    p_check_email.set_defaults(func=cmd_check_email)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
