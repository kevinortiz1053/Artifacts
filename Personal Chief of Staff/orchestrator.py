#!/usr/bin/env python3
"""
LLM orchestrator for the chore tracker / chief-of-staff assistant.
---------------------------------------------------------------------
Supervisor -> {Chores, Goals, General} -> (loops back to Supervisor) -> Vibe.

Entry point: chores.py's route_message() calls handle_message() here for
EVERY incoming email once an API key is configured (see chores.py's
route_message docstring for why -- in short, the supervisor has to see
every message before any domain-specific regex is allowed to claim it, so
a goals-related email containing a word like "done" can't get silently
misrouted as a chore command). With no API key configured, none of this
module is touched at all -- chores.py's handle_incoming_email() handles
everything with its own regex + help text, unchanged from before this
module existed.

Design notes:
- The Supervisor (Haiku -- fast/cheap) replaces what used to be a plain
  one-shot Router. It first DECOMPOSES a message into one or more steps,
  each assigned to exactly one specialist ('chores', 'goals', or
  'general'). Most messages are one step, handled exactly like before. A
  message that genuinely spans domains (e.g. "mark the trash chore done
  AND update my workout goal to on track") becomes a multi-step plan.
- A multi-step plan is NOT executed right away -- it's staged as a
  pending_plans.json entry (same pattern as chores.py's
  pending_deletions.json) and a preview is emailed back asking for
  confirmation, specifically so a bad split can be corrected before any
  specialist is actually prompted. A single-step "plan" needs no
  confirmation and runs immediately, same latency as before. The
  confirmation round trip is handled the same way chores.py's
  propose-then-confirm delete flow is: entirely by matching yes/no/other
  against the next reply, checked before any LLM call for that email --
  see get_pending_plan()/set_pending_plan() and the top of
  supervisor_node(). A reply that's neither yes nor no is treated as
  feedback and the plan gets revised and re-proposed rather than either
  blindly executed or ignored.
- Once a plan is confirmed (or is a single step to begin with), the graph
  actually cycles: each specialist node routes back to "supervisor"
  (not straight to "vibe" the way it used to) after it runs. The
  supervisor judges (one more small Haiku call) whether that step was
  actually completed; if not, it can send the SAME specialist another,
  refined attempt at the SAME step -- up to MAX_HOPS_PER_SPECIALIST (4)
  attempts total for that specialist across the whole request, after
  which it gives up on that step, notes as much, and moves on rather than
  looping forever. Once every step in the plan has been visited, the
  supervisor routes to "vibe" with all the steps' results combined.
- The Chores node tries chore_lib.try_regex_command() first -- the same
  deterministic, free fast path chores.py itself uses in degraded mode --
  before falling back to a tool-using ReAct agent for anything that isn't
  a plain done/list/add/remove command. This is what keeps plain chore
  commands cheap. It always acts on state["current_instruction"] -- the
  supervisor's per-step slice of the request, not necessarily the whole
  raw email -- which for a single-step plan is just the original message
  verbatim, so the fast path is exactly as available as it always was.
- The Goals node is always a tool-using agent (no regex fast path): every
  new goal needs an LLM-drafted plan, so there's no meaningfully "free"
  path the way there is for chores. Its tools call directly into goals.py's
  existing, already-tested functions, same pattern as the chores tools.
  Its tools are scoped per-request to who's actually asking (see
  resolve_requester/build_goals_tools) -- a household member emailing from
  their own address only sees/acts on their own goals by default.
- Deleting a chore is supported via a confirm-before-delete round trip
  (propose_delete_chore stages it, a follow-up "yes"/"no" reply -- handled
  entirely outside the LLM, by chores.py's route_message -- confirms or
  cancels it). Deleting a goal outright is still not exposed as a tool;
  use 'abandon' or the CLI's 'remove' for that.
- The General node is READ-ONLY, on purpose, and always will be: it has
  list_chores and goals' scoped list_goals/goal_detail, and NO write
  tools at all -- not because of a prompt instruction (those can be
  misread), but because the write tools simply aren't in the Python list
  passed to its agent, so there's no way for it to add/modify/delete
  anything even if a message or a bad plan step asked it to. Only the
  Chores and Goals specialists can write. This matters more now than it
  used to: the supervisor can assign a lookup step to 'general' as part
  of a larger plan, and it needs to be structurally impossible for that
  to double as a backdoor write path.
- The Vibe node is a final pass that rewrites whatever the specialists
  produced (a plan preview awaiting confirmation, a cancellation, or the
  combined results of every step) into a warm, concise reply, without
  inventing or dropping any concrete facts (chore/goal names, ids, dates,
  milestone text) or softening a "reply yes/no" call to action.
- Threading: this module never sends email itself -- it only returns a
  string from handle_message(), and chores.py's check_email_once() is
  what actually calls send_reply_email() with in_reply_to/references
  taken from whatever incoming message it's currently processing. That
  means EVERY reply this module produces -- including a plan preview, a
  cancellation, a "still waiting to hear back" nudge, and the final
  combined result -- automatically lands in the same email thread the
  triggering message was part of, with no special-casing needed here.
  The one thing that deliberately does NOT reply in-thread is
  chores.py's/goals.py's cmd_remind(), which calls send_email() (no
  in_reply_to/references at all) to start a fresh thread for each
  proactive reminder -- that's already how it works and nothing about
  this module changes it.
- Graphs are cached per API key at module level, since watch_inbox.py is a
  long-running process — we don't want to rebuild the graph on every email.
  Only the underlying LLM CLIENTS are cached, though -- each node builds
  its own tool list and agent fresh per request (see build_chores_tools /
  build_goals_tools), since several tools need to know who's actually
  asking.
"""

import json
import os
from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

import chores as chore_lib
import goals as goal_lib

ROUTER_MODEL = "claude-haiku-4-5-20251001"
AGENT_MODEL = "claude-sonnet-4-6"
VIBE_MODEL = "claude-sonnet-4-6"

SPECIALISTS = ("chores", "goals", "general")
MAX_HOPS_PER_SPECIALIST = 4

PENDING_PLANS_FILE = os.path.join(chore_lib.BASE_DIR, "pending_plans.json")
PENDING_PLAN_TTL_HOURS = 24


# --------------------------------------------------------------------------
# Chores tools — thin wrappers around chores.py's existing, already-tested
# functions (unchanged from before the goals node existed)
# --------------------------------------------------------------------------

@tool
def add_chore(name: str, assignee: str, type: str, due: str = "",
              frequency: str = "", days: str = "", day_of_month: str = "",
              interval_days: str = "", start: str = "") -> str:
    """Add a new chore.

    Args:
        name: The chore's name/description.
        assignee: Who is responsible for it.
        type: 'recurring' or 'adhoc'.
        due: Due date as MM-DD-YYYY. Required if type is 'adhoc'.
        frequency: 'daily', 'weekly', 'monthly', or 'interval'. Required if type is 'recurring'.
        days: Comma-separated weekday names (e.g. 'Monday, Thursday'). Required if frequency is 'weekly'.
        day_of_month: Day of the month, 1-31. Optional for 'monthly' (defaults to 1).
        interval_days: Number of days between occurrences. Required if frequency is 'interval'.
        start: Start date as MM-DD-YYYY for interval chores. Optional (defaults to today).
    """
    data = chore_lib.load_data()
    fields = {
        "name": name, "assignee": assignee, "type": type,
        "due": due or None, "frequency": frequency or None, "days": days or None,
        "day_of_month": day_of_month or None, "interval_days": interval_days or None,
        "start": start or None,
    }
    try:
        chore = chore_lib.build_chore_from_fields(data, fields)
    except ValueError as e:
        return f"Could not add chore: {e}"
    chore_lib.save_data(data)
    pos = chore_lib.display_position(data, chore)
    return f"Added chore #{pos}: {chore['name']} (assigned to {chore['assignee']})"


@tool
def list_chores(assignee: str = "") -> str:
    """Show the current chore status: what's overdue, due, and done.
    Optionally pass an assignee name to filter to just that person's chores."""
    data = chore_lib.load_data()
    return chore_lib.build_status_report(data, assignee=assignee or None)


@tool
def mark_done(chore_position: int) -> str:
    """Mark a chore as completed today, by its current position number (from
    list_chores' output -- overdue first, then due, then done). Note this
    position is recomputed fresh each time, not a stable id -- always call
    list_chores first if there's any chance the numbering has changed since
    you last saw it (e.g. earlier in the same conversation)."""
    data = chore_lib.load_data()
    c = chore_lib.mark_chore_done(data, chore_position)
    if not c:
        return f"No chore at position {chore_position}. Call list_chores to see current numbers."
    chore_lib.save_data(data)
    return f"Marked '{c['name']}' as done today."


CHORES_TOOLS = [add_chore, list_chores, mark_done]


def build_chores_tools(sender):
    """Per-request chores tool list: everything in CHORES_TOOLS (stateless,
    reusable across requests, no sender needed) plus a delete-proposal tool
    closed over THIS request's sender, since staging a pending deletion has
    to know who to ask for confirmation (see chores.py's propose_deletion /
    handle_pending_deletion_reply). Rebuilt fresh per request -- cheap,
    since it's just Python closures, no extra API call."""

    @tool
    def propose_delete_chore(items: str) -> str:
        """Propose deleting one or more EXISTING chores. This does NOT
        delete anything yet -- it stages a confirmation request and returns
        a preview of exactly what would be removed. The actual deletion
        only happens if the person replies confirming it in a follow-up
        email. ALWAYS call list_chores first so you're proposing chores
        that actually exist, using their exact name and assignee text as
        shown there -- never guess or delete on the first message alone.

        Args:
            items: One or more chores to propose deleting, one per line,
                each formatted exactly as 'name | assignee' (e.g.
                'Take out trash | Bob'). Use the exact name and assignee
                text from list_chores' output.
        """
        parsed = []
        for line in items.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, _, assignee = line.partition("|")
            name, assignee = name.strip(), assignee.strip()
            if name and assignee:
                parsed.append({"name": name, "assignee": assignee})
        if not parsed:
            return ("Couldn't parse any chores to propose deleting -- format each as "
                    "'name | assignee', one per line, using exact text from list_chores.")
        data = chore_lib.load_data()
        preview, _matched = chore_lib.propose_deletion(data, sender, parsed)
        chore_lib.save_data(data)
        return preview

    return CHORES_TOOLS + [propose_delete_chore]


# --------------------------------------------------------------------------
# Goals tools — thin wrappers around goals.py's existing, already-tested
# functions, same pattern as the chores tools above. Built per-request (see
# build_goals_tools) rather than once at module load, because every tool
# here needs to know WHO is asking: a household member emailing from their
# own address is scoped by default to their own goals, while the account
# owner (or any sender we don't recognize -- see resolve_requester) has
# full access to everyone's.
# --------------------------------------------------------------------------

def build_goals_tools(full_access, requester_assignee):
    """Per-request goals tool list, scoped to what this requester is
    allowed to see/do.

    full_access=True (the account owner, or a sender resolve_requester
    doesn't recognize) -- tools behave unrestricted, same as before this
    scoping existed.

    full_access=False, requester_assignee=<name> (a household member
    emailing from their own address, matched via config['assignee_emails'])
    -- every WRITE tool, and goal_detail, are hard-restricted in CODE to
    goals owned by that person: ownership is checked directly against the
    goal's assignee field, never left to the LLM's judgment, so a
    misunderstanding can't mutate or reveal detail on someone else's goal.
    list_goals is the one exception: it defaults to the requester's own
    goals too, but exposes a show_everyone flag the LLM can set when the
    message explicitly asks to see everyone's goals -- a read-only roster
    view is low-risk to share within a household, unlike a write."""

    def check_owns(goal):
        if full_access:
            return True
        return (goal.get("assignee") or "").strip().lower() == (requester_assignee or "").strip().lower()

    @tool
    def add_goal(name: str, assignee: str = "", description: str = "", target: str = "",
                 checkin_days: str = "") -> str:
        """Add a new goal. Returns the new goal's id -- follow this up with
        save_plan to draft and attach a plan, since every new goal should
        get one right away.

        Args:
            name: Short name for the goal.
            assignee: Who this goal belongs to. Required if you have full
                access (ask who it's for if it isn't clear) -- ignored (and
                automatically set to the requester) if you're scoped to one
                person, since they can only ever create goals for themselves.
            description: Optional longer description of what success looks like.
            target: Optional target date, MM-DD-YYYY or YYYY-MM-DD.
            checkin_days: Optional whole number of days between proactive
                check-in emails for this goal specifically, if the person asked
                for a particular cadence (e.g. "check in on me every 3 days",
                "remind me weekly" -> 7, "monthly" -> 30). Leave blank to use
                the default cadence.
        """
        if full_access:
            effective_assignee = assignee.strip()
            if not effective_assignee:
                return "Which household member is this goal for? Let me know and I'll add it."
        else:
            effective_assignee = requester_assignee

        data = goal_lib.load_data()
        fields = {
            "name": name, "assignee": effective_assignee, "description": description or None,
            "target": target or None, "checkin_interval": checkin_days or None,
        }
        try:
            goal = goal_lib.build_goal_from_fields(data, fields)
        except ValueError as e:
            return f"Could not add goal: {e}"
        goal_lib.save_data(data)
        return f"Added goal #{goal['id']}: {goal['name']} (assigned to {goal['assignee']})"

    @tool
    def set_checkin_frequency(goal_id: int, days: str = "") -> str:
        """Set how often (in days) an EXISTING goal gets a proactive check-in
        email, overriding the default cadence. Pass an empty `days` to reset
        the goal back to the default cadence instead of a custom one.

        Args:
            goal_id: The goal's numeric id.
            days: Whole number of days between check-ins (e.g. 3, 7, 14).
                Leave blank to reset to the default.
        """
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if goal is None:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't change it."
        try:
            goal = goal_lib.set_checkin_interval(data, goal_id, int(days) if days else None)
        except ValueError as e:
            return f"Could not set check-in frequency: {e}"
        goal_lib.save_data(data)
        if days:
            return f"#{goal['id']} '{goal['name']}' will now get a check-in every {days} day(s)."
        return f"#{goal['id']} '{goal['name']}' reset to the default check-in cadence."

    @tool
    def save_plan(goal_id: int, plan: str, milestones: str) -> str:
        """Attach a plan and milestone list to a goal. Call this right after
        add_goal for a brand-new goal -- draft the plan and milestones yourself
        first (you're the one who should write them, not the person emailing),
        then save them here.

        Args:
            goal_id: The goal's numeric id (from add_goal's response or list_goals).
            plan: A short paragraph describing the overall approach.
            milestones: Comma-separated concrete milestone steps, e.g.
                'Research programs, Pick one, Finish week 1, Finish week 2'.
        """
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if goal is None:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't update its plan."
        milestone_list = [m.strip() for m in milestones.split(",") if m.strip()]
        goal = goal_lib.set_plan(data, goal_id, plan, milestone_list)
        goal_lib.save_data(data)
        return f"Saved plan for #{goal['id']} '{goal['name']}' with {len(milestone_list)} milestone(s)."

    @tool
    def list_goals(status: str = "", assignee: str = "", show_everyone: bool = False) -> str:
        """Show current goals and their progress. Optionally filter by
        status: 'active', 'completed', or 'abandoned'.

        Args:
            status: Optional status filter.
            assignee: Optional person's name to filter to just their goals.
                Only meaningful if you have full access (ignored otherwise --
                a scoped requester's list is already limited to their own
                goals unless show_everyone is set).
            show_everyone: Set True ONLY if the message explicitly asks to
                see everyone's / the whole household's goals. Otherwise
                leave False -- a scoped requester's goals are shown by
                default, never everyone's, unless they actually asked.
        """
        data = goal_lib.load_data()
        if full_access or show_everyone:
            return goal_lib.build_status_report(data, status_filter=status or None, assignee=assignee or None)
        return goal_lib.build_status_report(data, status_filter=status or None, assignee=requester_assignee)

    @tool
    def goal_detail(goal_id: int) -> str:
        """Show full detail for one goal by id -- its description, plan,
        milestones, and recent check-ins."""
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if not goal:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't share its details."
        return goal_lib.build_goal_detail(goal)

    @tool
    def mark_milestone_done(goal_id: int, milestone_number: int) -> str:
        """Mark a milestone complete, by the goal's numeric id and the
        milestone's 1-based number (as shown in goal_detail's or list_goals'
        output)."""
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if goal is None:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't update it."
        goal, text = goal_lib.mark_milestone_done(data, goal_id, milestone_number)
        if not goal:
            return f"No goal/milestone found for goal {goal_id}, milestone {milestone_number}."
        goal_lib.save_data(data)
        return f"Marked milestone {milestone_number} done for #{goal['id']} '{goal['name']}': {text}"

    @tool
    def log_checkin(goal_id: int, note: str) -> str:
        """Log a check-in / progress note against a goal, by its numeric id."""
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if goal is None:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't log against it."
        goal = goal_lib.log_checkin(data, goal_id, note)
        goal_lib.save_data(data)
        return f"Logged check-in for #{goal['id']} '{goal['name']}'."

    @tool
    def set_goal_status(goal_id: int, status: str) -> str:
        """Mark a goal 'completed' or 'abandoned', by its numeric id."""
        data = goal_lib.load_data()
        goal = goal_lib.find_goal(data, goal_id)
        if goal is None:
            return f"No goal with id {goal_id}."
        if not check_owns(goal):
            return "That goal isn't assigned to you, so I can't change its status."
        try:
            goal = goal_lib.set_status(data, goal_id, status)
        except ValueError as e:
            return f"Could not update status: {e}"
        goal_lib.save_data(data)
        return f"Marked #{goal['id']} '{goal['name']}' as {status}."

    return [add_goal, save_plan, list_goals, goal_detail, mark_milestone_done,
            log_checkin, set_goal_status, set_checkin_frequency]


# --------------------------------------------------------------------------
# Requester scoping — who is this message from, and what are they allowed
# to see/do? Shared by goals_node (full scoping), general_node's read-only
# lookups, and chores_node's delete-proposal tool (which just needs a
# sender to stage the pending deletion against).
# --------------------------------------------------------------------------

def resolve_requester(sender, config):
    """Figure out what a message sender is allowed to see/do, based on
    config["assignee_emails"] (the same {name: email} directory chores.py
    and goals.py both use for per-person reminder routing).

    Returns (full_access, requester_assignee):
      - The account owner (sender matches config["sender_email"]), or any
        sender whose address isn't in assignee_emails at all (fail-open --
        e.g. no directory configured, or an address that isn't in it but
        was still allowed through check_email_once()'s allow list), gets
        full_access=True, requester_assignee=None: sees/acts on everything.
      - A sender whose address matches a specific assignee_emails entry
        gets full_access=False, requester_assignee=<that name>: scoped by
        default to their own goals (see build_goals_tools)."""
    sender_norm = (sender or "").strip().lower()
    owner = (config.get("sender_email") or "").strip().lower()
    if not sender_norm or sender_norm == owner:
        return True, None

    directory = config.get("assignee_emails") or {}
    reverse = {addr.strip().lower(): name for name, addr in directory.items()}
    assignee = reverse.get(sender_norm)
    if assignee is None:
        return True, None

    return False, assignee


# --------------------------------------------------------------------------
# Pending plan confirmation — same pattern as chores.py's
# pending_deletions.json (see set_pending_deletion/get_pending_deletion/
# clear_pending_deletion there), just for a multi-step supervisor plan
# instead of a chore deletion. Keyed by lowercased sender address, stored
# next to chores.json (chore_lib.BASE_DIR) so it lives alongside every
# other piece of this app's state rather than introducing a new location.
# Deliberately stores the STEPS (specialist + instruction) rather than
# anything that could go stale, since nothing about a plan's steps changes
# shape between proposal and confirmation the way a chore's position/id
# can -- unlike pending deletions, there's no re-resolution step needed.
# --------------------------------------------------------------------------

def load_pending_plans():
    if not os.path.exists(PENDING_PLANS_FILE):
        return {}
    with open(PENDING_PLANS_FILE, "r") as f:
        return json.load(f)


def save_pending_plans(state):
    with open(PENDING_PLANS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def set_pending_plan(sender, original_request, steps):
    state = load_pending_plans()
    state[(sender or "").strip().lower()] = {
        "original_request": original_request,
        "steps": steps,
        "proposed_at": datetime.now().isoformat(),
    }
    save_pending_plans(state)


def get_pending_plan(sender):
    """Returns the pending-plan entry for `sender`, or None if there isn't
    one or it's expired (older than PENDING_PLAN_TTL_HOURS). An expired
    entry is cleared as a side effect. Mirrors chores.py's
    get_pending_deletion() exactly."""
    state = load_pending_plans()
    key = (sender or "").strip().lower()
    entry = state.get(key)
    if entry is None:
        return None
    proposed_at = datetime.fromisoformat(entry["proposed_at"])
    age_hours = (datetime.now() - proposed_at).total_seconds() / 3600
    if age_hours > PENDING_PLAN_TTL_HOURS:
        clear_pending_plan(sender)
        return None
    return entry


def clear_pending_plan(sender):
    state = load_pending_plans()
    state.pop((sender or "").strip().lower(), None)
    save_pending_plans(state)


# --------------------------------------------------------------------------
# Structured-output schemas for the supervisor's two small LLM judgment
# calls: splitting a message into steps, and checking whether a step a
# specialist just ran actually got finished.
# --------------------------------------------------------------------------

class PlanStep(BaseModel):
    specialist: str = Field(
        description="Which specialist handles this step: exactly one of "
                     "'chores', 'goals', or 'general'.")
    instruction: str = Field(
        description="A complete, self-contained instruction for that "
                     "specialist covering just this one piece of the "
                     "request -- it won't see the original message, only "
                     "this text, so include any names/dates/details it "
                     "needs.")


class RequestPlan(BaseModel):
    steps: list[PlanStep] = Field(
        description="One or more steps needed to fully address the "
                     "request, in the order they should be carried out.")


class StepJudgment(BaseModel):
    done: bool = Field(
        description="True if the specialist's response shows this step's "
                     "instruction was fully carried out (or, for a pure "
                     "lookup, that it actually answered using real data "
                     "rather than guessing).")
    retry_instruction: str = Field(
        default="",
        description="Only if done=False and another attempt by the SAME "
                     "specialist could plausibly finish it: a short, "
                     "refined instruction to retry with. Leave blank if "
                     "done, or if retrying wouldn't help (e.g. it's "
                     "genuinely missing information only the person "
                     "emailing could supply) -- in that case set done=True "
                     "instead so the reply goes back to them rather than "
                     "being retried pointlessly.")


DECOMPOSE_SYSTEM = (
    "You are the planning supervisor for a household chief-of-staff assistant. Read "
    "the incoming message and break it into one or more steps, each assigned to "
    "exactly one specialist:\n\n"
    "- 'chores' -- adding, completing, or deleting household chores/tasks.\n"
    "- 'goals' -- setting a new personal goal, updating a plan, logging progress, "
    "checking off a milestone, or changing a goal's status or check-in cadence. "
    "Household chores are NOT goals even when phrased similarly -- a recurring task "
    "like 'water the plants' is a chore; a longer-term personal objective like 'get "
    "back in shape' or 'learn Spanish' is a goal.\n"
    "- 'general' -- READ-ONLY lookups (listing/checking existing chores or goals), "
    "plus small talk, research requests (not supported yet), or anything else that "
    "doesn't fit chores/goals. 'general' can NEVER make a change -- it has no "
    "ability to add, complete, delete, or update anything at all, so never assign an "
    "action that modifies data to 'general'; that always belongs to 'chores' or "
    "'goals' instead.\n\n"
    "Most messages need just ONE step -- don't split a message that's really about "
    "one thing into multiple pieces. Only produce more than one step when the "
    "message genuinely asks for separate actions in separate domains (e.g. 'mark "
    "the trash chore done AND update my workout goal to on track' is two steps: one "
    "chores, one goals). Each step's instruction must be a complete, self-contained "
    "request that specialist can act on without seeing the original message. If the "
    "message is really just one thing, return exactly one step, with the "
    "instruction being the message itself (verbatim, or only lightly cleaned up)."
)

JUDGE_SYSTEM = (
    "You are checking whether a specialist assistant fully completed the ONE step it "
    "was just given, as part of a larger multi-step request. Read the step's "
    "instruction and the specialist's response. Mark done=True if the response shows "
    "the action was completed, or -- for a pure lookup -- that it actually answered "
    "using real data. Mark done=False only if the response shows the specialist was "
    "blocked or asked a clarifying question that a rephrased/more specific "
    "instruction could plausibly resolve on retry, and in that case write that "
    "refined instruction. If retrying wouldn't actually help (e.g. it's genuinely "
    "missing information only the person emailing could provide), mark done=True "
    "anyway so the response goes back to them instead of being retried pointlessly."
)


def decompose_request(planner_llm, body, feedback=None):
    """Ask the LLM to split `body` into one or more specialist-assigned
    steps. If `feedback` is given, this is a REVISION of an earlier
    proposed plan the person didn't just confirm outright -- see
    supervisor_node()'s pending-plan branch. Always returns at least one
    step; falls back to a single 'general' step covering the whole
    message if the model somehow returns nothing usable."""
    prompt = f"Message:\n{body}"
    if feedback:
        prompt += (
            f"\n\nYou already proposed a plan for this message and the person replied "
            f"with feedback instead of a plain yes/no: \"{feedback}\". Revise the plan "
            f"to account for what they said."
        )
    structured_llm = planner_llm.with_structured_output(RequestPlan)
    result = structured_llm.invoke([SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=prompt)])
    steps = []
    for s in (result.steps if result and result.steps else []):
        specialist = (s.specialist or "").strip().lower()
        if specialist not in SPECIALISTS:
            specialist = "general"
        instruction = (s.instruction or "").strip() or body
        steps.append({"specialist": specialist, "instruction": instruction})
    if not steps:
        steps = [{"specialist": "general", "instruction": body}]
    return steps


def judge_step_completion(judge_llm, instruction, output):
    """One cheap Haiku call: did the specialist actually finish this step?
    Returns {"done": bool, "retry_instruction": str}."""
    prompt = f"Step instruction:\n{instruction}\n\nSpecialist's response:\n{output}"
    structured_llm = judge_llm.with_structured_output(StepJudgment)
    result = structured_llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)])
    return {"done": bool(result.done), "retry_instruction": (result.retry_instruction or "").strip()}


def format_plan_preview(steps):
    labels = {"chores": "Chores", "goals": "Goals", "general": "Look up"}
    lines = [
        "That touches more than one thing, so here's how I'd split it up -- I want "
        "to check this is right before doing anything:",
        "",
    ]
    for i, step in enumerate(steps, start=1):
        label = labels.get(step["specialist"], step["specialist"].title())
        lines.append(f"{i}. {label}: {step['instruction']}")
    lines.append("")
    lines.append("Reply 'yes' to go ahead, 'no' to cancel, or tell me what to change.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------

class GraphState(TypedDict):
    email_body: str
    sender: str
    route: str
    current_instruction: str
    node_output: str
    final_reply: str
    plan: list
    step_index: int
    hop_counts: dict
    step_outputs: list
    original_request: str


# --------------------------------------------------------------------------
# Graph construction (cached per API key — see module docstring)
# --------------------------------------------------------------------------

_graph_cache = {}


def _build_graph(api_key):
    # Only the underlying LLM clients are cached/reused across requests --
    # the AGENTS themselves (chores_agent, goals_agent) are built fresh
    # INSIDE chores_node/goals_node on every call, from per-request tool
    # lists scoped to that request's sender (see build_chores_tools /
    # build_goals_tools / resolve_requester). That's cheap: create_react_agent
    # just wires a client to a tool list in Python, no extra API call --
    # only invoking the agent costs anything.
    #
    # NOTE: langgraph.prebuilt.create_react_agent is deprecated as of
    # LangGraph 1.0 in favor of langchain.agents.create_agent -- but that
    # replacement lives in the top-level `langchain` package, which
    # requires Python 3.10+ and won't install at all on a Python 3.9
    # environment (confirmed against LangChain's own migration docs after
    # this bit us on a real deployment). Staying on create_react_agent
    # deliberately for now; it's not actually deprecated in the pre-1.0
    # langgraph/langchain-core line requirements.txt's floors allow, which
    # is what a Python 3.9 install will actually resolve to anyway. Revisit
    # this once the deployment target is on Python 3.10+.
    planner_llm = ChatAnthropic(model=ROUTER_MODEL, temperature=0, api_key=api_key)
    vibe_llm = ChatAnthropic(model=VIBE_MODEL, temperature=0.4, api_key=api_key)
    chores_llm = ChatAnthropic(model=AGENT_MODEL, temperature=0, api_key=api_key)
    goals_llm = ChatAnthropic(model=AGENT_MODEL, temperature=0.2, api_key=api_key)
    general_llm = ChatAnthropic(model=AGENT_MODEL, temperature=0.3, api_key=api_key)

    def supervisor_node(state: GraphState) -> dict:
        plan = state.get("plan") or []
        sender = state["sender"]
        body = state["email_body"]

        if not plan:
            # Nothing executing yet -- either this is a brand-new request,
            # or it's a reply to a plan preview we sent earlier and are
            # waiting to hear back on.
            pending = get_pending_plan(sender)
            if pending is not None:
                text = chore_lib.strip_quoted_reply(body)

                if chore_lib.looks_like_cancellation(text):
                    clear_pending_plan(sender)
                    return {
                        "route": "vibe",
                        "node_output": "Okay, I won't do that -- nothing was changed.",
                        "plan": [], "step_index": 0, "hop_counts": {}, "step_outputs": [],
                    }

                if chore_lib.looks_like_confirmation(text):
                    clear_pending_plan(sender)
                    steps = pending["steps"]
                    first = steps[0]
                    return {
                        "route": first["specialist"], "plan": steps, "step_index": 0,
                        "hop_counts": {}, "step_outputs": [],
                        "current_instruction": first["instruction"],
                        "original_request": pending["original_request"],
                    }

                # Neither a clear yes nor no -- treat it as feedback on the
                # plan itself and revise it, rather than executing a plan
                # we were never actually told to run, or nagging for a
                # plain yes/no when they just told us what to fix.
                revised = decompose_request(planner_llm, pending["original_request"], feedback=text)
                if len(revised) <= 1:
                    first = revised[0]
                    clear_pending_plan(sender)
                    return {
                        "route": first["specialist"], "plan": revised, "step_index": 0,
                        "hop_counts": {}, "step_outputs": [],
                        "current_instruction": first["instruction"],
                        "original_request": pending["original_request"],
                    }
                preview = format_plan_preview(revised)
                set_pending_plan(sender, pending["original_request"], revised)
                return {
                    "route": "vibe", "node_output": preview,
                    "plan": [], "step_index": 0, "hop_counts": {}, "step_outputs": [],
                }

            # Brand-new request.
            steps = decompose_request(planner_llm, body)
            if len(steps) <= 1:
                first = steps[0]
                return {
                    "route": first["specialist"], "plan": steps, "step_index": 0,
                    "hop_counts": {}, "step_outputs": [],
                    "current_instruction": first["instruction"],
                    "original_request": body,
                }
            preview = format_plan_preview(steps)
            set_pending_plan(sender, body, steps)
            return {
                "route": "vibe", "node_output": preview,
                "plan": [], "step_index": 0, "hop_counts": {}, "step_outputs": [],
            }

        # Returning here after a specialist ran one step of an
        # already-confirmed (or single-step) plan -- decide whether that
        # step is actually done, or needs another attempt by the same
        # specialist (up to MAX_HOPS_PER_SPECIALIST total for it across
        # this whole request), then move on to the next step or finish.
        step_index = state.get("step_index", 0)
        hop_counts = dict(state.get("hop_counts") or {})
        step_outputs = list(state.get("step_outputs") or [])
        current = plan[step_index]
        specialist = current["specialist"]
        output = state.get("node_output", "")

        hop_counts[specialist] = hop_counts.get(specialist, 0) + 1
        judgment = judge_step_completion(planner_llm, current["instruction"], output)
        can_retry = hop_counts[specialist] < MAX_HOPS_PER_SPECIALIST

        if not judgment["done"] and judgment["retry_instruction"] and can_retry:
            new_plan = list(plan)
            new_plan[step_index] = {**current, "instruction": judgment["retry_instruction"]}
            return {
                "route": specialist, "plan": new_plan, "step_index": step_index,
                "hop_counts": hop_counts, "step_outputs": step_outputs,
                "current_instruction": judgment["retry_instruction"],
            }

        labels = {"chores": "Chores", "goals": "Goals", "general": "Lookup"}
        label = labels.get(specialist, specialist.title())
        note = output
        if not judgment["done"]:
            note += "\n\n(Couldn't fully resolve this after a few tries.)"
        step_outputs.append(f"{label}: {note}")

        next_index = step_index + 1
        if next_index >= len(plan):
            combined = "\n\n".join(step_outputs)
            return {
                "route": "vibe", "node_output": combined,
                "plan": [], "step_index": 0, "hop_counts": hop_counts, "step_outputs": step_outputs,
            }

        next_step = plan[next_index]
        return {
            "route": next_step["specialist"], "plan": plan, "step_index": next_index,
            "hop_counts": hop_counts, "step_outputs": step_outputs,
            "current_instruction": next_step["instruction"],
        }

    def chores_node(state: GraphState) -> dict:
        instruction = state.get("current_instruction") or state["email_body"]

        # Fast path first: the supervisor has already decided this step is
        # about chores, so it's now safe to apply the same deterministic
        # regex chores.py itself uses in degraded mode -- no LLM call needed
        # for a plain "done 3" or "list". For a single-step plan (the
        # overwhelming majority of messages), `instruction` is the original
        # message verbatim, so this is exactly as available as before.
        data = chore_lib.load_data()
        fast = chore_lib.try_regex_command(instruction, data)
        if fast is not None:
            reply, changed = fast
            if changed:
                chore_lib.save_data(data)
            return {"node_output": reply}

        # Not a plain regex command (natural language) -- fall back to the
        # tool-using agent. Built fresh per request with tools scoped to
        # this sender (see build_chores_tools) -- cheap, no extra API call.
        chores_tools = build_chores_tools(state["sender"])
        system = (
            "You are the chore-management assistant for a household chore tracker. "
            "You may be handling just ONE PIECE of a larger request the person made "
            "-- the instruction below is your complete scope; don't worry about "
            "anything else they may have asked for elsewhere. Use the available "
            "tools to fulfill it. "
            "When adding a chore, confidently translate plain language into "
            "add_chore's fields yourself rather than asking the person to restate "
            "it in field syntax — that's your job, not theirs. For example: 'due "
            "every Thursday' or 'every week on Thursday' means type=recurring, "
            "frequency=weekly, days=Thursday; 'every day' means frequency=daily; "
            "'due 3-15-2027' or a specific one-time date means type=adhoc with "
            "that due date; 'every 2 weeks' means frequency=interval, "
            "interval_days=14. Only ask a clarifying question when something is "
            "genuinely missing or ambiguous — no name at all, no assignee at all, "
            "or a cadence you can't confidently map to daily/weekly/monthly/"
            "interval/a specific date — never for something you can reasonably "
            "infer. A chore can only have ONE assignee — if a request names more "
            "than one person for the same chore (e.g. 'for Kevin and Veronica'), "
            "call add_chore once per person (same name, same schedule, one chore "
            "each) rather than combining the names into a single assignee field; "
            "this is what makes per-person reminder routing work for each of "
            "them. Mention in your reply that you split it into one chore per "
            "person. If a request also asks about WHEN reminder emails go out "
            "(e.g. 'send a reminder every Thursday morning'), note that you can't "
            "create or change an email schedule yourself — reminders go out "
            "whenever `chores.py remind` is already scheduled to run (e.g. via "
            "cron), and this chore will automatically be included in that once "
            "it's due/overdue; if they want a specific day/time for reminders "
            "and aren't sure their schedule covers it, say so honestly rather "
            "than implying you've set up a new schedule. To delete a chore, call "
            "list_chores first to find the exact chore(s), then "
            "propose_delete_chore with their exact name and assignee — this only "
            "STAGES the deletion and returns a preview, it does not delete "
            "anything yet. The person has to reply confirming it in a follow-up "
            "email before it actually happens (that confirmation reply is "
            "handled outside of you, automatically) — so once you've called "
            "propose_delete_chore, just relay its preview back to them as your "
            "answer; don't call it a second time and don't claim the chore is "
            "already deleted."
        )
        chores_agent = create_react_agent(chores_llm, tools=chores_tools)
        result = chores_agent.invoke({"messages": [
            SystemMessage(content=system),
            HumanMessage(content=instruction),
        ]})
        last_message = result["messages"][-1]
        return {"node_output": last_message.content}

    def goals_node(state: GraphState) -> dict:
        # Unlike chores, there's no meaningfully "free" fast path here: even
        # a plain-looking "add goal: learn guitar" needs an LLM to draft the
        # plan/milestones, so this always goes through the tool-using agent.
        # Tools (and the scope_note below) are built fresh per request,
        # scoped to who this message is actually from -- see
        # resolve_requester / build_goals_tools.
        instruction = state.get("current_instruction") or state["email_body"]
        config = chore_lib.load_config()
        full_access, requester_assignee = resolve_requester(state["sender"], config)
        goals_tools = build_goals_tools(full_access, requester_assignee)

        if full_access:
            scope_note = (
                "You have full access -- you can see and manage every household "
                "member's goals. When adding a new goal, always confirm/ask who "
                "it's for if it isn't clear from the message."
            )
        else:
            scope_note = (
                f"This message is from {requester_assignee}. By default, only show "
                f"and act on {requester_assignee}'s OWN goals -- new goals they "
                f"create are automatically assigned to them, and you cannot modify "
                f"or reveal detail on anyone else's goal even if asked (the tools "
                f"themselves enforce this, so don't try to work around it). The one "
                f"exception: if their message explicitly asks to see EVERYONE's / "
                f"the whole household's goals (not just their own), call list_goals "
                f"with show_everyone=True for that one read -- otherwise leave it "
                f"False."
            )

        system = (
            "You are the goal-tracking and accountability assistant for a personal "
            "chief-of-staff. You may be handling just ONE PIECE of a larger request "
            "the person made -- the instruction below is your complete scope; don't "
            "worry about anything else they may have asked for elsewhere. Use the "
            f"available tools to fulfill it. {scope_note} "
            "When someone describes a NEW goal, call add_goal, then immediately draft a "
            "short, concrete plan (a couple of sentences) and 3-6 milestones "
            "yourself, and save them with save_plan before you respond — don't wait "
            "to be asked for a plan, that's expected every time a goal is created. "
            "Every active goal gets a proactive check-in email on a default cadence "
            "unless told otherwise — if the person specifies (or implies) how often "
            "they want to be checked in on, e.g. 'check in on me every 3 days', "
            "'remind me weekly' (= 7), 'monthly nudges' (= 30), set that cadence: "
            "pass checkin_days to add_goal for a brand-new goal, or call "
            "set_checkin_frequency for an existing one. Don't set a custom cadence "
            "unless they actually asked for one — the default is fine otherwise. "
            "Confirm the cadence you set in your reply so it's clear what was "
            "configured. For questions about existing goals, use list_goals or "
            "goal_detail to see the current state before answering. Be precise with "
            "goal ids and milestone numbers — if you're not sure which goal someone "
            "means, ask a clarifying question instead of guessing. You cannot delete "
            "a goal outright yet — if asked, explain that and suggest 'abandon' "
            "instead, or the CLI's 'remove' command."
        )
        goals_agent = create_react_agent(goals_llm, tools=goals_tools)
        result = goals_agent.invoke({"messages": [
            SystemMessage(content=system),
            HumanMessage(content=instruction),
        ]})
        last_message = result["messages"][-1]
        return {"node_output": last_message.content}

    def general_node(state: GraphState) -> dict:
        # This node exists for anything that doesn't cleanly fit chores or
        # goals alone -- pure lookups spanning both, small talk, or a
        # read-only step of a larger supervisor-split plan. It is READ-ONLY
        # BY CONSTRUCTION: the tool list below has no write tools in it at
        # all, so there's no way for this node to add/modify/delete
        # anything no matter what a message or a bad plan step asks for --
        # only chores_node and goals_node get write tools. It used to have
        # NO tools at all, which meant a request like "list all my chores
        # and goals" got a plausible-sounding but entirely made-up answer
        # ("looks like nothing's set up yet!") since the model had no way
        # to actually check -- a real bug found from a live email.
        instruction = state.get("current_instruction") or state["email_body"]
        config = chore_lib.load_config()
        full_access, requester_assignee = resolve_requester(state["sender"], config)
        goals_tools_all = build_goals_tools(full_access, requester_assignee)
        goals_list_tool = next(t for t in goals_tools_all if t.name == "list_goals")
        goal_detail_tool = next(t for t in goals_tools_all if t.name == "goal_detail")
        general_tools = [list_chores, goals_list_tool, goal_detail_tool]
        system = (
            "You are a helpful household chief-of-staff assistant. You may be "
            "handling just ONE PIECE of a larger request the person made -- the "
            "instruction below is your complete scope; don't worry about anything "
            "else they may have asked for elsewhere. Chores and goal-tracking are "
            "fully wired up; web research is coming soon but not available yet. You "
            "have READ-ONLY tools to look up current chores and goals -- use them "
            "whenever a request could involve real data (e.g. 'what's going on', "
            "'list everything', 'show me my chores and goals', or anything that "
            "implies checking a specific thing) rather than guessing. NEVER state or "
            "imply anything about what chores or goals exist -- including that there "
            "are none -- without having actually called the relevant tool first; an "
            "honest 'let me check' is always better than a confident guess. You "
            "cannot add, complete, delete, or update a chore or goal -- you have no "
            "tools that do that at all, on purpose. If asked to make any change, say "
            "so plainly rather than pretending to do it (the supervisor should "
            "normally route changes to the chores/goals specialists instead of you, "
            "but say so anyway if one slips through). If the request is clearly a "
            "research request, say so honestly rather than pretending to do it. "
            "Otherwise, just be a helpful, friendly assistant."
        )
        general_agent = create_react_agent(general_llm, tools=general_tools)
        result = general_agent.invoke({"messages": [
            SystemMessage(content=system),
            HumanMessage(content=instruction),
        ]})
        last_message = result["messages"][-1]
        return {"node_output": last_message.content}

    def vibe_node(state: GraphState) -> dict:
        system = (
            "You rewrite a draft assistant response into the final email reply. "
            "Keep every concrete fact exactly as given — chore/goal names, ids, "
            "dates, milestone text, counts — never invent or drop details. If the "
            "draft asks the person to reply confirming, cancelling, or changing "
            "something (e.g. 'reply yes to confirm', 'reply no to cancel'), always "
            "preserve that call to action clearly — never drop or soften it. Make "
            "the tone warm, concise, and encouraging. Do not add a greeting or a "
            "sign-off; just the body."
        )
        prompt = f"Draft response to rewrite:\n\n{state['node_output']}"
        resp = vibe_llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        return {"final_reply": resp.content}

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("chores", chores_node)
    graph.add_node("goals", goals_node)
    graph.add_node("general", general_node)
    graph.add_node("vibe", vibe_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda state: state["route"],
        {"chores": "chores", "goals": "goals", "general": "general", "vibe": "vibe"},
    )
    # Every specialist loops back to the supervisor instead of going
    # straight to vibe -- this is what makes the multi-step plan loop
    # possible (see supervisor_node's second half).
    graph.add_edge("chores", "supervisor")
    graph.add_edge("goals", "supervisor")
    graph.add_edge("general", "supervisor")
    graph.add_edge("vibe", END)

    return graph.compile()


def _get_graph(api_key):
    if api_key not in _graph_cache:
        _graph_cache[api_key] = _build_graph(api_key)
    return _graph_cache[api_key]


def handle_message(body, sender, api_key):
    """Run one email body through the orchestrator graph and return the final
    reply text."""
    graph = _get_graph(api_key)
    result = graph.invoke({
        "email_body": body,
        "sender": sender or "",
        "route": "",
        "current_instruction": "",
        "node_output": "",
        "final_reply": "",
        "plan": [],
        "step_index": 0,
        "hop_counts": {},
        "step_outputs": [],
        "original_request": "",
        # A multi-step plan with the 4-hop-per-specialist retry budget can
        # legitimately need more graph steps than LangGraph's default
        # recursion_limit (25) allows -- raise it well above the realistic
        # worst case (a handful of steps x up to 4 hops each, plus the
        # supervisor visit between every one of them).
    }, config={"recursion_limit": 100})
    return result["final_reply"]
