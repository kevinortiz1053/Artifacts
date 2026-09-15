#!/usr/bin/env python3
"""
Show exactly what chores.json check-email is actually reading, and hunt for
other copies of chores.json on disk that might be the "real" one with your
data in it.

Run this the SAME WAY check-email actually runs (same user, same cron/
systemd context, same working directory) for the results to be meaningful --
just like diagnose_llm_path.py.

Place this file in the same folder as chores.py before running.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chores  # noqa: E402

print(f"Running as user:     {os.environ.get('USER') or os.environ.get('USERNAME')}")
print(f"Current directory:   {os.getcwd()}")
print(f"chores.py location:  {chores.__file__}")
print(f"BASE_DIR:            {chores.BASE_DIR}")
print(f"DATA_FILE path:      {chores.DATA_FILE}")
print(f"DATA_FILE exists:    {os.path.exists(chores.DATA_FILE)}")
print()

if os.path.exists(chores.DATA_FILE):
    stat = os.stat(chores.DATA_FILE)
    print(f"Last modified:       {time.ctime(stat.st_mtime)}")
    print(f"Size:                {stat.st_size} bytes")
    data = chores.load_data()
    print(f"Chore count:         {len(data['chores'])}")
    print(f"next_id:             {data['next_id']}")
    if data["chores"]:
        print()
        print("Chores in this file:")
        for c in data["chores"]:
            print(f"  #{c['id']:<3} {c['name']:<25} -> {c['assignee']:<10} "
                  f"({c['type']}, last_completed={c.get('last_completed')})")
    else:
        print()
        print("DIAGNOSIS: this file exists but has zero chores -- either you")
        print("added chores somewhere else, or this file got reset.")
else:
    print("DIAGNOSIS: no chores.json at this path at all. load_data() is")
    print("silently returning an empty structure ({'chores': [], 'next_id': 1})")
    print("-- that's exactly why 'list' says 'No chores yet.' with no error.")

print()
print("--- Searching for other chores.json files on this machine ---")
print("(This helps spot a second copy of the project with your real data.)")
search_roots = [os.path.expanduser("~"), "/opt", "/srv", "/home"]
found = {}  # keyed by realpath, so a file reachable via two overlapping
            # search roots (e.g. ~ and /home, when ~ is /home/someuser)
            # only gets reported once instead of showing up as a fake
            # "duplicate".
for root in search_roots:
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        # don't descend into noisy/irrelevant trees
        dirnames[:] = [d for d in dirnames if d not in (
            ".git", "node_modules", "venv", ".venv", "__pycache__", "site-packages"
        )]
        if "chores.json" in filenames:
            p = os.path.join(dirpath, "chores.json")
            try:
                real = os.path.realpath(p)
                st = os.stat(p)
                found[real] = (p, st.st_size, st.st_mtime)
            except OSError:
                pass

found_list = list(found.values())
if not found_list:
    print("No chores.json files found under: " + ", ".join(search_roots))
else:
    for p, size, mtime in sorted(found_list, key=lambda x: -x[2]):
        same = " <-- this is the one check-email reads" if os.path.realpath(p) == os.path.realpath(chores.DATA_FILE) else ""
        print(f"  {p}  ({size} bytes, modified {time.ctime(mtime)}){same}")
    if len(found_list) > 1:
        print()
        print("Multiple DISTINCT chores.json files found. Whichever one has")
        print("your real data needs to be next to the SAME chores.py that")
        print("check-email actually runs -- either move/symlink it there, or")
        print("point your cron/systemd job at that directory instead.")
