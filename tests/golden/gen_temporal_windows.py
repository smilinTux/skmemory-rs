#!/usr/bin/env python3
"""Generate the temporal-window golden: Python `parse_time_window(strict=True)` over a large query set.

Usage (reads Python skmemory, writes only tests/golden/temporal_windows.json):

    PYTHONPATH=<skmemory checkout> python tests/golden/gen_temporal_windows.py

`now` is fixed at 2026-11-01T12:00:00Z (the bench clock). The query set covers every phrase family
of Python's parser (ISO dates and months, month plus day, month plus year, bare years, since/before/
between, relative days/weeks/months, last/past N, "the past month", weekdays, this/last week/month,
ambiguous may/march) alone, in contexts, with punctuation and unicode variants, and in pairs.
The Rust test (`tests/temporal_parity.rs`) requires: for every query Rust returns EXACTLY Python's
window or abstains, never a different window. Regenerate, never hand-edit.
"""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from skmemory.fusion import parse_time_window

HERE = Path(__file__).resolve().parent
NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

REL = [
    "yesterday", "today", "Yesterday", "TODAY", "yesterday!", "today?", "x_yesterday", "yesterday_x",
    "re-yesterday", "éyesterday", "yesterdays",
    "3 days ago", "1 day ago", "two weeks ago", "a month ago", "an hour ago", "12 months ago",
    "999 days ago", "1000 days ago", "0 days ago", "5 weeks ago", "eleven months ago",
    "5-days-ago", "5, days ago", "5   days   ago", "5\tdays\nago", "5 days, ago", "5 dayss ago",
    "5 days agox", "x5 days ago", "5_days ago", "٥ days ago", "5 days ago ago",
    "last 3 days", "past 2 weeks", "in the last 5 months", "over the past 7 days", "last 1 month",
    "last twelve weeks", "last a day", "past  3  days", "last 3 daysx", "last 3-days", "last, 3 days",
    "last 31 months", "last 400 days",
    "the past month", "past week", "past day", "past month.", "past month and", "past month, then",
    "past week or two", "past day to", "past month but", "past month from", "past month we",
    "past month i", "past month the", "past months", "past month", "past month-wide", "past month x",
    "past week?", "past day;",
    "last monday", "last Friday", "last sunday", "last  tuesday", "last-monday", "last wednesday.",
    "last mondays", "lastmonday", "last thursday and last saturday",
    "last week", "this week", "last week of october", "this week of june", "this week, of x",
    "this week   of x", "this week of", "last weeks", "this  week", "this week-end",
    "last month", "this month", "last month of the year", "this months", "last\nmonth",
]
CAL = [
    "in june", "in June", "on june", "since october", "before december", "after april", "from july",
    "during august", "until september", "by february", "till november", "in the january", "in the may",
    "in may", "May 5", "May", "march", "March outage", "in march", "the rollout may halt", "we march 4 hours",
    "I may go", "in sept", "in jun", "since oct", "in dec", "in december 2025", "in june 2025",
    "since june 2025", "in june 5", "on june 5", "june 5", "5 june", "5th of june", "in june 5th",
    "in 2026", "in 2025", "1999", "2030", "in 2026-10", "2026-10-16", "2026-10", "2026-13-01",
    "2026-02-30", "on 2026-10-16", "since 2026-10", "Q3 2026", "October 2025", "oct 2025",
    "between june and august", "between june 3 and july 9", "from june to august",
    "in early june", "in the month of june", "back in june", "as of june", "around june", "mid june",
    "late june", "throughout june", "june", "june notes", "the June outage", "what did we decide june 5",
    "step 3 may fail", "5 may", "5 may 2026", "5th of may", "may 2026", "in may 2026", "in march 2026",
    "march 4", "sometime in the june release", "june and july", "in june or july", "in june, july",
    "since april in june", "June release notes in july", "after the may release in june",
    "in jun 2025", "from may to june", "in june-july", "in june/july", "in-june", "in june",
    "in  june", "in the  june", "inJune", "in junes", "in june's", "im june", "in june.", "(in june)",
    "in éjune", "in juneé", "in june 2", "in 5 june", "in 20 june",
]
CTX = [("", ""), ("what did we decide ", ""), ("Show ", " please"), ("", "?"), ("notes from ", " about the deploy")]


def window(q):
    try:
        r = parse_time_window(q, now=NOW, strict=True)
    except Exception as e:  # a Python exception is recorded; Rust must abstain
        return {"error": type(e).__name__}
    return list(r) if r else None


queries = []
seen = set()


def add(q):
    if q not in seen:
        seen.add(q)
        queries.append(q)


for phrase in REL + CAL:
    for pre, post in CTX:
        add(f"{pre}{phrase}{post}")
for a, b in itertools.product(REL[::3] + CAL[::4], REL[1::4] + CAL[1::5]):
    add(f"{a} and {b}")
    add(f"{b} then {a}")
fixtures = Path(HERE.parent / "fixtures" / "eval-v1" / "queries-v1.jsonl")
for line in fixtures.read_text().splitlines():
    if line.strip():
        add(json.loads(line)["query"])
out = [{"q": q, "w": window(q)} for q in queries]
(HERE / "temporal_windows.json").write_text(json.dumps(out, ensure_ascii=True, separators=(",", ":")) + "\n")
n = sum(1 for r in out if isinstance(r["w"], list))
print(f"{len(out)} queries, python answers {n}, abstains {sum(1 for r in out if r['w'] is None)}, errors {sum(1 for r in out if isinstance(r['w'], dict))}")
