#!/usr/bin/env python3
"""Generate the Python-written parity vectors for the intention store, tick and ledger (SKMEM-R-07).

Usage:
    PYTHONPATH=<skmemory checkout or export> python tests/intent_parity/gen_intent_parity.py

Everything here is produced by the real Python modules (`skmemory.intentions`,
`skmemory.intention_tick`, `skmemory.ledger`, `skmemory.ledger_sidecar`) and
written under `tests/intent_parity/`:

* `py_store/`   an intention store as Python writes it (flat files; the index is derived);
* `py_conflict/` a store with Syncthing conflict copies, for the merge on reindex;
* `py_ledger/`  a ledger as Python writes it, with a torn tail, a duplicate and an id-less line;
* `expected.json` what Python did with them: dumps, transitions, merges, prefilter and tick
  runs (fired ids, the file bytes and the `intentions` rows afterwards), ledger events,
  scores and appended bytes.

The Rust tests (`tests/intent_parity.rs`) load the same inputs and must reproduce every
value. Synthetic data only. Regenerate, never hand-edit.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from skmemory import intention_tick as tick
from skmemory import intentions as I
from skmemory import ledger as L
from skmemory import ledger_sidecar as S

HERE = Path(__file__).resolve().parent
FIX = HERE.parent / "fixtures" / "eval-v1"
NOW = "2026-11-01T12:00:00+00:00"


def dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def snapshot(base: Path) -> dict:
    d = base / "intentions"
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.json"))}
    con = sqlite3.connect(str(base / "index.db"))
    try:
        rows = [list(r) for r in con.execute("SELECT * FROM intentions ORDER BY id")]
    finally:
        con.close()
    return {"files": files, "rows": rows}


def delta(snap: dict, pristine: dict) -> dict:
    """Only what differs from the pristine snapshot (keeps expected.json small)."""
    return {
        "files": {k: v for k, v in snap["files"].items() if pristine["files"].get(k) != v},
        "deleted": sorted(set(pristine["files"]) - set(snap["files"])),
        "rows": [r for r in snap["rows"] if r not in pristine["rows"]],
        "row_ids": [r[0] for r in snap["rows"]],
    }


def fresh(src: Path, tmp: Path, name: str) -> Path:
    dst = tmp / name
    shutil.copytree(src, dst)
    return dst


def jl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_store(base: Path):
    """Fixture intentions plus edge intentions, written by Python."""
    if base.exists():
        shutil.rmtree(base)
    st = I.IntentionStore(base)
    for row in jl(FIX / "intentions-v1.jsonl"):
        st.save(I.Intention(**row))
    edge = [
        dict(id="e-unicode", kind="event", title="Café ☃ \U0001f600 \"q\" \\ \t tab", state="armed",
             trigger_terms=["Café au lait", "STRASSE", "xß"], scope={"zrepo": "a", "arepo": "b"},
             created_at="2026-10-01T08:30:15.250Z", max_fires=3, cooldown_s=0, origin_memory_id="m-3c059f37b4",
             note="extra field", weight=1.5, nested={"a": [1, 2, {"b": None}], "e": []}),
        dict(id="e-offset", kind="event", title="Offset times", state="armed", trigger_terms=["offset probe"],
             created_at="2026-10-01T10:00:00+05:30", expires_at="2027-01-01T00:00:00-08:00",
             last_fired_at="2026-10-02T00:00:00.000123+02:00", fires=1, cooldown_s=60),
        dict(id="e-nonword", kind="event", title="No word chars", state="armed", trigger_terms=["=>", "(*)"],
             created_at="2026-10-01T10:00:00Z"),
        dict(id="e-multi", kind="event", title="Multi word", state="armed", trigger_terms=["Alpha Beta", "gamma"],
             created_at="2026-10-01T10:00:00Z", max_fires=5),
        dict(id="e-many1", kind="event", title="Fire cap 1", state="armed", trigger_terms=["capword"],
             created_at="2026-10-01T10:00:00Z"),
        dict(id="e-many2", kind="event", title="Fire cap 2", state="armed", trigger_terms=["capword"],
             created_at="2026-10-01T10:00:00Z"),
        dict(id="e-many3", kind="event", title="Fire cap 3", state="armed", trigger_terms=["capword"],
             created_at="2026-10-01T10:00:00Z"),
        dict(id="e-many4", kind="event", title="Fire cap 4", state="armed", trigger_terms=["capword"],
             created_at="2026-10-01T10:00:00Z"),
        dict(id="t-repeat", kind="time", title="Repeating", state="armed", due_at="2026-10-31T12:00:00Z",
             interval_s=86400, created_at="2026-10-01T10:00:00Z", origin_memory_id="m-449a549be1"),
        dict(id="t-limited", kind="time", title="Limited repeat", state="armed", due_at="2026-10-31T12:00:00Z",
             interval_s=3600, max_fires=1, created_at="2026-10-01T10:00:00Z"),
        dict(id="t-ctrl", kind="time", title="Ctrl\x07 chars\nand nbsp ​ zwsp " + "x" * 260,
             state="pending", due_at="2026-10-30T12:00:00Z", created_at="2026-10-01T10:00:00Z",
             origin_memory_id="mé\ttab"),
        dict(id="t-later", kind="time", title="Due later", state="pending", due_at="2026-11-02T00:00:00Z",
             created_at="2026-10-01T10:00:00Z"),
        dict(id="t-snooze", kind="time", title="Snoozed", state="armed", due_at="2026-10-30T12:00:00Z",
             snooze_until="2026-11-01T13:00:00Z", created_at="2026-10-01T10:00:00Z"),
    ]
    for e in edge:
        st.save(I.Intention(**e))
    st.reindex()
    st.close()


def origin_files(base: Path):
    d = base / "long-term"
    d.mkdir(parents=True, exist_ok=True)
    for r in jl(FIX / "corpus-valid-to-v1.jsonl"):
        (d / f"{r['id']}.json").write_text(json.dumps(r), encoding="utf-8")
    (base / "mid-term").mkdir(exist_ok=True)
    (base / "mid-term" / "m-meta.json").write_text(
        json.dumps({"id": "m-meta", "metadata": {"valid_to": "2026-10-01T00:00:00Z"}}), encoding="utf-8")


def build_conflict(base: Path):
    if base.exists():
        shutil.rmtree(base)
    st = I.IntentionStore(base)
    d = base / "intentions"
    mk = lambda **k: I.Intention(**{"kind": "event", "title": "T", "trigger_terms": ["x"], "created_at": "2026-10-01T10:00:00Z", **k})
    st.save(mk(id="c-live", state="armed"))
    # replicas of one id: armed vs fired vs done on other hosts
    (d / "c-live.sync-conflict-20261101-120000-AAAAAAA.json").write_text(
        mk(id="c-live", state="fired", fires=2, last_fired_at="2026-10-20T10:00:00Z").model_dump_json(indent=2), encoding="utf-8")
    (d / "c-live.sync-conflict-20261101-120100-BBBBBBB.json").write_text(
        mk(id="c-live", state="fired", fires=3, last_fired_at="2026-10-21T10:00:00Z", title="Other").model_dump_json(indent=2), encoding="utf-8")
    st.save(mk(id="c-dead", state="armed"))
    (d / "c-dead.sync-conflict-20261101-120000-CCCCCCC.json").write_text(
        mk(id="c-dead", state="cancelled").model_dump_json(indent=2), encoding="utf-8")
    # conflict only: promoted to canonical
    (d / "c-only.sync-conflict-20261101-120000-DDDDDDD.json").write_text(
        mk(id="c-only", state="armed").model_dump_json(indent=2), encoding="utf-8")
    # file whose stem differs from its id
    (d / "stem-mismatch.json").write_text(mk(id="c-mismatch", state="pending").model_dump_json(indent=2), encoding="utf-8")
    # unreadable
    (d / "bad.json").write_text("{not json", encoding="utf-8")
    (d / "badkind.json").write_text(json.dumps({"id": "bk", "kind": "nope", "title": "x"}), encoding="utf-8")
    st.close()
    for p in base.glob("index.db*"):
        p.unlink()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="intent-parity-"))
    out = {"fixture_now": NOW}
    try:
        # ---- stores written by Python ----
        store_dir = HERE / "py_store"
        build_store(store_dir)
        origin_files(store_dir)
        for p in store_dir.glob("index.db*"):
            p.unlink()  # the index is derived: Rust rebuilds it
        conflict_dir = HERE / "py_conflict"
        build_conflict(conflict_dir)

        pristine = tmp / "pristine"
        shutil.copytree(store_dir, pristine)
        st = I.IntentionStore(pristine)
        st.reindex()
        st.close()
        P = snapshot(pristine)
        out["reindex_rows"] = P["rows"]

        # ---- dumps ----
        raws = [
            {"id": "d1", "kind": "time", "title": "t", "created_at": "2026-10-01T10:00:00Z", "due_at": "2026-11-06T12:00:00+02:00"},
            {"id": "d2", "kind": "event", "title": "ünï ☃ \u0000 \u001f", "created_at": "2026-11-06T12:00:00.5Z",
             "x1": 1.0, "x2": 1e21, "x3": 1e-7, "x4": 100000.0, "x5": [1, 2.5, None, True], "x6": {}, "x7": 12345678901234567890},
            {"id": "d3", "kind": "event", "title": "t", "created_at": "2026-10-01T10:00:00Z", "cooldown_s": None, "fires": None, "scope": {"b": "1", "a": "2"}},
            {"id": "d4", "kind": "time", "title": "t", "created_at": "2026-11-06 12:00:00", "due_at": "2026-11-06",
             "snooze_until": "2026-11-06T12:00:00.123456789+00:00"},
            {"id": "d5", "kind": "time", "title": "t", "created_at": "2026-10-01T10:00:00Z", "interval_s": 5.0, "max_fires": 3.0},
        ]
        out["dumps"] = []
        for r in raws:
            text = json.dumps(r)
            i = I.Intention(**json.loads(text))
            out["dumps"].append({"input": text, "indent2": i.model_dump_json(indent=2), "compact": i.model_dump_json()})
        bad = [
            {"id": "v1", "kind": "nope", "title": "t"}, {"id": "v2", "kind": "time"}, {"kind": "time", "title": "t"},
            {"id": "v3", "kind": "time", "title": "t", "state": None}, {"id": "v4", "kind": "event", "title": "t", "trigger_terms": [1]},
            {"id": "v5", "kind": "event", "title": "t", "scope": {"a": 1}}, {"id": "v6", "kind": "time", "title": "t", "due_at": "junk"},
            {"id": "v7", "kind": "time", "title": "t", "channel": None}, {"id": "v8", "kind": "time", "title": "t", "fires": 1.5},
            {"id": "v9", "kind": "time", "title": "t", "state": "bogus"}, {"id": "v10", "kind": "time", "title": "t", "interval_s": "x"},
        ]
        out["invalid"] = []
        for r in bad:
            try:
                I.Intention(**r)
                raise SystemExit(f"expected invalid: {r}")
            except SystemExit:
                raise
            except Exception:
                out["invalid"].append(json.dumps(r))

        # ---- transitions ----
        base_t = dict(id="x", kind="time", title="T", due_at="2026-11-01T12:00:00Z", created_at="2026-10-01T10:00:00Z")
        cases = []

        def tcase(over, event, now, snooze=None):
            i = I.Intention(**{**base_t, **over})
            c = {"intention": i.model_dump_json(), "event": event, "now": now, "snooze": snooze}
            try:
                r = I.transition(i, event, now=dt(now) if now else None, snooze_until=dt(snooze) if snooze else None)
                c["result"] = r.model_dump_json(indent=2)
            except ValueError as e:
                c["error"] = str(e)
            cases.append(c)

        for s in ["pending", "armed", "fired", "done", "cancelled", "expired", "invalidated"]:
            for e in ["arm", "fire", "done", "snooze", "cancel", "expire", "invalidate", "bogus", ""]:
                tcase({"state": s}, e, NOW)
        tcase({"state": "armed"}, "snooze", NOW, "2026-11-05T00:00:00+02:00")
        tcase({"state": "fired", "snooze_until": "2026-11-02T00:00:00Z"}, "snooze", NOW)
        for now in ["2026-11-01T12:00:00Z", "2026-11-01T12:00:01Z", "2026-11-01T13:00:00Z", "2026-11-01T12:59:59.999999Z",
                    "2026-11-11T12:30:00Z", "2026-10-01T12:00:00Z", "2026-11-01T12:00:00+03:00", "2027-11-01T12:00:00Z"]:
            tcase({"state": "armed", "interval_s": 3600}, "fire", now)
            tcase({"state": "pending", "interval_s": 86400, "snooze_until": "2026-11-01T00:00:00Z"}, "fire", now)
        tcase({"state": "armed", "interval_s": 3600, "fires": 2, "max_fires": 3}, "fire", NOW)
        tcase({"state": "armed", "interval_s": 3600, "fires": 1, "max_fires": 3}, "fire", NOW)
        tcase({"state": "armed", "interval_s": 0}, "fire", NOW)
        tcase({"state": "armed", "interval_s": -5}, "fire", NOW)
        tcase({"state": "armed", "kind": "event", "interval_s": 60, "trigger_terms": ["a"]}, "fire", NOW)
        tcase({"state": "armed", "due_at": "2026-11-01T12:00:00+02:00", "interval_s": 7200}, "fire", "2026-11-01T14:00:00Z")
        out["transitions"] = cases

        # ---- merges ----
        mk = lambda **k: I.Intention(**{"kind": "event", "title": "T", "id": "m", "created_at": "2026-10-01T10:00:00Z", **k})
        pairs = [
            (mk(state="armed"), mk(state="fired", fires=1, last_fired_at="2026-10-02T00:00:00Z")),
            (mk(state="done"), mk(state="cancelled")), (mk(state="cancelled"), mk(state="invalidated")),
            (mk(state="invalidated"), mk(state="expired")), (mk(state="expired"), mk(state="fired", fires=9)),
            (mk(state="fired", fires=2, last_fired_at="2026-10-02T00:00:00Z"), mk(state="fired", fires=2, last_fired_at="2026-10-03T00:00:00+02:00", title="B")),
            (mk(state="fired", fires=2, title="A"), mk(state="fired", fires=2, title="B")),
            (mk(state="fired", fires=1, last_fired_at="2026-10-05T00:00:00Z"), mk(state="fired", fires=3, last_fired_at="2026-10-02T00:00:00Z")),
            (mk(state="armed", title="same"), mk(state="armed", title="same")),
            (mk(state="pending", z=1), mk(state="pending", y=2)),
        ]
        out["merges"] = []
        for a, b in pairs:
            ab, ba = I.merge_intentions(a, b), I.merge_intentions(b, a)
            assert ab == ba
            out["merges"].append({"a": a.model_dump_json(), "b": b.model_dump_json(), "merged": ab.model_dump_json(indent=2)})

        # ---- prefilter ----
        pf = []

        def pcase(text, scope=None, now=NOW, env=None, kwargs=None):
            for k in list(os.environ):
                if k.startswith("SKMEMORY_PREFILTER_"):
                    del os.environ[k]
            for k, v in (env or {}).items():
                os.environ[f"SKMEMORY_PREFILTER_{k}"] = v
            base = fresh(store_dir, tmp, f"pf{len(pf)}")
            s = I.IntentionStore(base)
            s.reindex()
            fired = s.prefilter(text, scope, dt(now), **(kwargs or {}))
            s.close()
            snap = delta(snapshot(base), P)
            pf.append({"text": text, "scope": scope or {}, "now": now, "env": env or {}, "kwargs": kwargs or {},
                       "fired": [f.id for f in fired], **snap})
            for k in list(os.environ):
                if k.startswith("SKMEMORY_PREFILTER_"):
                    del os.environ[k]

        for p in jl(FIX / "prompts-v1.jsonl"):
            pcase(p["text"], p["scope"], p["now"])
        pcase("Café au lait please", {"zrepo": "a", "arepo": "b"})
        pcase("CAFÉ AU LAIT", {"zrepo": "a", "arepo": "b"})
        pcase("strasse xß", {"zrepo": "a", "arepo": "b"})
        pcase("the offset probe!", {}, now="2026-10-02T00:00:30+02:00")
        pcase("the offset probe!", {}, now="2026-10-02T00:02:00+02:00")
        pcase("the offset probe!", {}, now="2027-02-01T00:00:00Z")
        pcase("a => b (*) c")
        pcase("alpha beta", {}); pcase("alpha  beta"); pcase("alpha\tbeta"); pcase("xalpha beta"); pcase("alpha betas"); pcase("GAMMA ray")
        pcase("capword here")
        pcase("capword here", {}, kwargs={"max_fires": 2})
        pcase("capword here", {}, env={"MAX_FIRES": "1"})
        pcase("capword here", {}, env={"MAX_FIRES": "0"})
        pcase("capword here", {}, kwargs={"deadline_s": 0.0})
        pcase("")
        pcase("   \n ")
        # head and tail scan, cut-token drop
        pcase("fusion " + "z" * 40, {"repo": "skmemory"}, env={"MAX_CHARS": "20"})
        pcase("pad " * 4 + "con" + "fusion" + " " * 6 + "end", {"repo": "skmemory"}, env={"MAX_CHARS": "32"})
        pcase("x" * 30 + " fusion", {"repo": "skmemory"}, env={"MAX_CHARS": "20"})
        pcase("fusion" + " " * 40, {"repo": "skmemory"}, env={"MAX_CHARS": "20"})
        pcase("a" * 8 + "con" + "fusion" + "b" * 8, {"repo": "skmemory"}, env={"MAX_CHARS": "20"})
        pcase("x" * 100 + " rollout" + " ring " + "y" * 100, {"repo": "skcapstone"}, env={"MAX_CHARS": "30"})
        pcase("rollout ring " + "q " * 50, {"repo": "skcapstone"}, env={"MAX_CHARS": "26"})
        pcase("q " * 50 + "rollout ring", {"repo": "skcapstone"}, env={"MAX_CHARS": "26"})
        pcase("q " * 50 + "ring", {"repo": "skcapstone"}, env={"MAX_CHARS": "27"})
        pcase("rollout" + "é" * 20 + " ring", {"repo": "skcapstone"}, env={"MAX_CHARS": "10"})
        pcase("fusion", {"repo": "skmemory"}, env={"MAX_CHARS": "nope", "DEADLINE_S": "x"})
        out["prefilter"] = pf

        # ---- sweep ----
        base = fresh(store_dir, tmp, "sweep")
        s = I.IntentionStore(base)
        s.reindex()
        changed = s.sweep_supersession(dt(NOW))
        s.close()
        out["sweep"] = {"changed": changed, **delta(snapshot(base), P)}
        base = fresh(store_dir, tmp, "sweep2")
        s = I.IntentionStore(base)
        s.reindex()
        changed = s.sweep_supersession(dt("2026-09-01T00:00:00Z"))
        s.close()
        out["sweep_early"] = {"changed": changed}

        # ---- conflict merge ----
        cbase = fresh(conflict_dir, tmp, "conf")
        s = I.IntentionStore(cbase)
        n = s.reindex()
        rep = s.last_reindex_report
        s.close()
        out["conflict"] = {"indexed": n, "report": {"indexed": rep["indexed"], "merged": sorted([[m["id"], sorted(m["files"])] for m in rep["merged"]]),
                           "conflicts": sorted(rep["conflicts"]), "unreadable": sorted(rep["unreadable"])}, **snapshot(cbase)}

        # ---- tick ----
        def run(label, deliver_results, **kw):
            base = fresh(store_dir, tmp, f"tick{len(out.get('tick', []))}")
            s = I.IntentionStore(base)
            sent = []
            it = iter(deliver_results)

            def deliver(t):
                sent.append(t)
                return next(it, True)

            now_s = kw.pop("now", NOW)
            r = tick.run_tick(s, deliver=deliver, any_host=True, now=dt(now_s), **kw)
            s.close()
            snap = delta(snapshot(base), P)
            out.setdefault("tick", []).append({"label": label, "results": deliver_results, "now": now_s, "kw": kw, "skipped": r.skipped,
                                               "fired": r.fired, "failed": r.failed, "remaining": r.remaining, "sent": sent, **snap})

        run("all-ok", [])
        run("cap-2", [], max_per_tick=2)
        run("fail-first", [False])
        run("fail-second", [True, False])
        run("summary-fails", [True, True, False], max_per_tick=1)
        run("later", [], now="2026-11-03T00:00:00Z")
        run("cap-0-is-1", [], max_per_tick=0)
        base = fresh(store_dir, tmp, "tickhost")
        s = I.IntentionStore(base)
        hostcases = []
        for want, host, any_host in [("SKAP08", "skap08.lan", False), ("skap08", "other", False), ("", "x", False), ("", "x", True), ("Straße", "STRASSE.local", False)]:
            r = tick.run_tick(s, deliver=lambda t: True, tick_host=want or None, hostname=host, any_host=any_host, now=dt("2020-01-01T00:00:00Z"))
            hostcases.append({"tick_host": want, "hostname": host, "any_host": any_host, "skipped": r.skipped})
        s.close()
        out["tick_hosts"] = hostcases
        out["alert_text"] = []
        for row in jl(FIX / "intentions-v1.jsonl") + [dict(id="t-ctrl", kind="time", title="Ctrl\x07 chars\nand nbsp ​ zwsp " + "x" * 260, origin_memory_id="mé\ttab"),
                                                        dict(id="t-short", kind="time", title="x" * 197 + "yyy"), dict(id="t-200", kind="time", title="y" * 200), dict(id="t-201", kind="time", title="y" * 201)]:
            out["alert_text"].append({"id": row["id"], "text": tick.alert_text(I.Intention(**row))})
        cps = ["\x00", "\x1f", " ", "~", "\x7f", "\x85", "\xa0", "\xad", "é", " ", "​", "﻿", "", "\U0001f600", "͸", "\U000e0001", "　", "힤"]
        out["printable"] = [[c, c.isprintable()] for c in cps if not (0xD800 <= ord(c) <= 0xDFFF)]
        out["shlex"] = []
        for s_ in ["sk-alert", "my cmd --flag 'a b'", 'x "a \\" b" c\\ d', "a 'unterminated", "", "  spaced   out ", 'q "a\\nb"', "a\\"]:
            try:
                out["shlex"].append({"in": s_, "out": __import__("shlex").split(s_)})
            except ValueError:
                out["shlex"].append({"in": s_, "out": None})

        # ---- parse_due_at ----
        out["parse_due_at"] = []
        now = dt(NOW)
        for v in ["in 5 minutes", "IN 2 Hours", "in 3 days", "in 1 week", "in 10mins", "in 1 hr", "in  7  days ", "  in 4 hrs\n",
                  "in 1000000 days", "in 0 days", "in 5 fortnights", "in five days", "2026-12-01T09:00:00Z", "2026-12-01T09:00:00",
                  "2026-12-01", " 2026-12-01T09:00:00+02:00 ", "garbage", "", "   ", "in 5", "in5 days"]:
            c = {"value": v}
            try:
                c["result"] = I.parse_due_at(v, now).isoformat()
            except ValueError as e:
                c["error"] = str(e)
            out["parse_due_at"].append(c)

        # ---- locks ----
        import hashlib
        out["sha1"] = [[x, hashlib.sha1(x.encode()).hexdigest()] for x in ["", "/a/b", "/home/u/.skcapstone/agents/x/memory", "café/☃"]]

        # ---- ledger ----
        build_ledger(HERE / "py_ledger", out)
        # ---- id regex ----
        out["ids"] = [[x, bool(I._ID_RE.match(x))] for x in ["a", "a1.b_c-d", "-a", ".a", "", "a/b", "a\n", "a\n\n", "A" * 128, "A" * 129, "a b", "é", "a\x00"]]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (HERE / "expected.json").write_text(json.dumps(out, indent=1, ensure_ascii=True, sort_keys=False) + "\n", encoding="utf-8")
    print("wrote", HERE / "expected.json")


def build_ledger(base: Path, out: dict):
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    host = "skap01"
    led = L.Ledger(base, host=host)
    calls = [
        ("recover-schema", "1.2", "served", dict(session="s1", at="2026-08-13T12:00:00+00:00", event_id="a" * 32)),
        ("recover-schema", "1.2", "followed", dict(effect="success", session="s1", at="2026-08-13T12:00:01+00:00", event_id="b" * 32)),
        ("recover-schema", "1.2", "followed", dict(effect="success", session="s1", at="2026-09-02T00:00:00.5+02:00", event_id="c" * 32, source="explicit:agent")),
        ("sync-to-clear-red", "1.0", "served", dict(session="s2", at="2026-09-30T23:59:59+00:00", event_id="d" * 32)),
        ("sync-to-clear-red", "1.0", "followed", dict(effect="harm", session="s2", at="2026-10-01T00:00:00+00:00", event_id="e" * 32)),
        ("sync-to-clear-red", "1.0", "followed", dict(effect="harm", session="s2", at="2026-10-02T00:00:00", event_id="f" * 32)),
        ("sync-to-clear-red", "1.0", "followed", dict(effect="harm", session="s2", at="2026-10-03T00:00:00Z", event_id="1" * 32)),
        ("sync-to-clear-red", "1.0", "followed_error", dict(session="s2", at="2026-10-04T00:00:00Z", event_id="2" * 32, source=S.SRC_ERROR, missing="cause unknown")),
        ("neutral-skill", "3", "followed", dict(effect="neutral", session="s3", at="2026-10-05T00:00:00Z", event_id="3" * 32)),
        ("neutral-skill", "3", "served", dict(session="s3", at="2026-10-05T00:00:00Z", event_id="4" * 32)),
        ("café-skill", "sha:abc", "served", dict(session="sé", at="2026-10-06T00:00:00Z", event_id="5" * 32, source="x" * 100, missing="m" * 300)),
        ("future-skill", "1", "followed", dict(effect="success", session="s", at="2027-01-01T00:00:00Z", event_id="6" * 32)),
        ("never-followed", "1", "served", dict(session="s", at="2026-10-07T00:00:00Z", event_id="7" * 32)),
        ("never-followed", "1", "served", dict(session="s", at="2026-10-08T00:00:00Z", event_id="8" * 32)),
        ("never-followed", "1", "served", dict(session="s", at="2026-10-09T00:00:00Z", event_id="9" * 32)),
        ("hermes-skill", "patch-2", "served", dict(session="hermes-usage", at="2026-10-09T00:00:00Z", event_id=S._sha("hermes", "hermes-skill", "served", "0"), source=S.SRC_HERMES)),
        ("hermes-skill", "patch-2", "followed", dict(session="hermes-usage", at="2026-10-09T00:00:00Z", event_id=S._sha("hermes", "hermes-skill", "followed", "0"), source=S.SRC_HERMES)),
        ("hermes-skill", "patch-2", "followed", dict(session="hermes-usage", at="2026-10-09T00:00:00Z", event_id=S._sha("hermes", "hermes-skill", "followed", "1"), source=S.SRC_HERMES)),
        ("hermes-skill", "patch-2", "followed", dict(session="hermes-usage", at="2026-10-09T00:00:00Z", event_id=S._sha("hermes", "hermes-skill", "followed", "2"), source=S.SRC_HERMES)),
    ]
    for skill, ver, ev, kw in calls:
        led.record(skill, ver, ev, **kw)
    led.record("recover-schema", "1.2", "served", session="s1", at="2026-08-13T12:00:00+00:00", event_id="a" * 32)  # known id: no new line
    out["ledger_calls"] = [{"skill": s, "version": v, "event": e, **k} for s, v, e, k in calls]
    out["ledger_files"] = {p.name: p.read_text(encoding="utf-8") for p in sorted((base / "ledger").glob("*.jsonl"))}
    # other host, a duplicate id, an id-less line, a torn line, junk, numeric version
    other = base / "ledger" / "outcomes-skwk02-2026-10.jsonl"
    lines = [
        json.dumps({"skill": "recover-schema", "version": "1.2", "event": "served", "at": "2026-08-13T12:00:00+00:00", "effect": None, "session": "s1", "id": "a" * 32}),
        json.dumps({"skill": "idless", "version": "1", "event": "followed", "at": "2026-10-10T00:00:00+00:00", "effect": "success", "session": "s9"}),
        json.dumps({"skill": "numver", "version": 2, "event": "served", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s9", "id": "n1"}),
        json.dumps({"skill": "floatver", "version": 1.5, "event": "served", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s9", "id": "n2"}),
        json.dumps({"skill": "nullver", "version": None, "event": "served", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s9", "id": "n3"}),
        json.dumps({"skill": "listver", "version": ["a", 1, None, True, {"k": 1.0}], "event": "served", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s9", "id": "n4"}),
        "not json at all",
        json.dumps({"skill": "badevent", "version": "1", "event": "bogus", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s"}),
        json.dumps({"skill": "badat", "version": "1", "event": "served", "at": "yesterday", "effect": None, "session": "s"}),
        json.dumps({"skill": "missing", "version": "1", "event": "served", "at": "2026-10-10T00:00:00Z"}),
        json.dumps([1, 2]),
        json.dumps({"skill": "emptyid", "version": "1", "event": "served", "at": "2026-10-10T00:00:00Z", "effect": None, "session": "s", "id": ""}),
        json.dumps({"skill": "noeffectkey", "version": "1", "event": "followed", "at": "2026-10-10T00:00:00Z", "session": "s", "id": "n5", "source": "", "missing": 5}),
        '{"skill": "torn", "version": "1", "event": "served", "at": "2026-10-1',
    ]
    other.write_text("\n".join(lines), encoding="utf-8")  # no trailing newline: a torn tail
    out["ledger_torn_before"] = other.read_text(encoding="utf-8")
    led2 = L.Ledger(base, host="skwk02")
    led2.record("after-torn", "1", "served", session="s", at="2026-10-11T00:00:00Z", event_id="t0" * 16)
    out["ledger_torn_file"] = other.read_text(encoding="utf-8")
    ev = L.Ledger(base, host=host).events()
    out["ledger_events"] = ev
    led.reindex()
    con = sqlite3.connect(str(base / "index.db"))
    out["ledger_rows"] = [list(r) for r in con.execute("SELECT * FROM ledger_events ORDER BY id")]
    con.close()
    (base / "index.db").unlink()
    out["ledger_scores"] = {}
    for as_of in ["2026-11-01T12:00:00+00:00", "2026-12-25T00:00:00+00:00", "2026-10-01T00:00:00+02:00"]:
        out["ledger_scores"][as_of] = [
            {**s.__dict__, "net_repr": repr(s.net), "follow_rate_repr": repr(s.follow_rate)} for s in led.scores(as_of)
        ]
    out["event_ids"] = [
        {"args": a, "id": L.event_id(*a)} for a in [
            ("s", "1", "served", "2026-10-10T00:00:00+00:00", None, "x"),
            ("café", "v☃", "followed", "2026-10-10T00:00:00+00:00", "success", "\U0001f600"),
            ('q"\\', "1", "followed", "t", "harm", "\n"),
        ]]
    out["sha"] = [{"parts": p, "id": S._sha(*p)} for p in [["a", "b"], ["hermes", "skill", "served", "0"], ["", ""], ["café", "x"]]]
    # hermes import
    usage = base / "usage.json"
    usage.write_text(json.dumps({
        "alpha": {"use_count": 2, "view_count": 1, "last_used_at": "2026-10-20T00:00:00Z", "last_viewed_at": "2026-10-19T00:00:00Z", "patch_count": 3},
        "beta": {"use_count": 0, "view_count": 3, "last_viewed_at": "2026-10-21T00:00:00Z"},
        "gamma": {"use_count": 0, "view_count": 0},
        "delta": "not a dict",
        "eps": {"use_count": "2", "created_at": "2026-10-22T00:00:00Z"},
    }, indent=1), encoding="utf-8")
    hb = base / "hermes"
    hl = L.Ledger(hb, host="skap01")
    st1 = S.import_hermes_usage(usage, hl)
    st2 = S.import_hermes_usage(usage, hl)
    out["hermes"] = {"first": st1, "second": st2, "events": hl.events(),
                     "files": {p.name: p.read_text(encoding="utf-8") for p in sorted((hb / "ledger").glob("*.jsonl"))}}
    shutil.rmtree(hb)



if __name__ == "__main__":
    main()
