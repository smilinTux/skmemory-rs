#!/usr/bin/env python3
"""Python loads what the Rust engine wrote (SKMEM-R-07).

Usage: PYTHONPATH=<skmemory> python tests/intent_parity/verify_rust_intent.py <base>

<base> holds `intentions/*.json`, `index.db` and `ledger/` written by the Rust
store. Checks, raising on the first failure:

* every flat file loads as `Intention` and re-dumps to the same bytes;
* Python opens Rust's `index.db` (its DDL and migrations must accept it), and the
  Rust-written `intentions` row of each file equals the row Python would write;
* Python's prefilter and list run on the Rust-written index;
* the ledger loads without skipping a line and Python's reindex reproduces the rows
  Rust indexed on append.
"""
import json
import sqlite3
import sys
from pathlib import Path

from skmemory import intentions as I
from skmemory import ledger as L

base = Path(sys.argv[1])
n = 0
bodies = {}
for p in sorted((base / "intentions").glob("*.json")):
    text = p.read_text(encoding="utf-8")
    i = I.Intention(**json.loads(text))
    assert i.model_dump_json(indent=2) == text, f"{p.name} is not byte-stable in Python"
    bodies[i.id] = (i, text)
    n += 1
assert n, "no intentions written"

con = sqlite3.connect(str(base / "index.db"))
rust_rows = {r[0]: tuple(r) for r in con.execute("SELECT * FROM intentions")}
con.close()
st = I.IntentionStore(base)  # runs Python's _ensure_table over Rust's table
for iid, (i, text) in bodies.items():
    want = I.IntentionStore._row(i, text)
    assert rust_rows[iid] == want, f"row {iid}: rust {rust_rows[iid]!r} != python {want!r}"
assert len(st.list()) == n
st.prefilter("anything at all", {})  # works on a Rust-written index
st.close()

led_dir = base / "ledger"
if led_dir.is_dir():
    host = sorted(led_dir.glob("outcomes-*.jsonl"))[0].name.split("-")[1]
    led = L.Ledger(base, host=host)
    lines = sum(1 for f in led_dir.glob("outcomes-*.jsonl") for ln in f.read_text("utf-8").splitlines() if ln.strip())
    ev = led.events()
    assert len(ev) == lines, f"python read {len(ev)} of {lines} lines"
    con = sqlite3.connect(str(base / "index.db"))
    before = sorted(tuple(r) for r in con.execute("SELECT * FROM ledger_events"))
    con.close()
    led.reindex()
    con = sqlite3.connect(str(base / "index.db"))
    after = sorted(tuple(r) for r in con.execute("SELECT * FROM ledger_events"))
    con.close()
    assert before == after, "rust-indexed ledger rows differ from python's reindex"
print("OK", n, "intentions")
