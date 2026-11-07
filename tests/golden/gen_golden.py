#!/usr/bin/env python3
"""Generate the golden files the Rust conformance test checks against.

Usage (reads Python skmemory, writes only under tests/golden/):

    PYTHONPATH=<skmemory checkout> python tests/golden/gen_golden.py \
        [--corpus <skmemory>/skmemory/eval/fixtures/corpus-v1.jsonl]

For each case it writes ``in/<name>.json`` (the stored record as a node would
find it on disk) and ``out/<name>.json`` (the exact bytes Python's
``FileBackend.save`` writes after ``Memory.from_stored``). ``manifest.json``
records the Python integrity hash and signature per case, which cases Python
refuses to load, and the tombstone bytes. Fixtures are generated, never
hand-edited: rerun this script instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skmemory.models import Memory

HERE = Path(__file__).resolve().parent
TS = "2026-10-02T12:00:00+00:00"


def base(**kw):
    d = {"id": "case", "title": "T", "content": "C", "created_at": TS, "updated_at": TS}
    d.update(kw)
    return d


CASES: dict[str, dict] = {}


def case(name, rec, raw=None):
    CASES[name] = rec


case(
    "full-v2-unicode",
    base(
        id="11111111-1111-4111-8111-111111111111",
        title="Café 日本語 \U0001f600 \"quoted\" back\\slash",
        content="line1\nline2\ttab\r\n\u0001ctl \u007fdel  sep ퟿ end",
        summary="s",
        tags=["b", "a", "ü"],
        layer="long-term",
        role="ai",
        source="session",
        source_ref="sess-1",
        emotional={
            "intensity": 8.5,
            "valence": 0.9,
            "labels": ["trust", "joy", "love"],
            "resonance_note": "felt ♥",
            "cloud9_achieved": True,
        },
        related_ids=["x", "y"],
        parent_id="p-1",
        context_tag="@work:chiro",
        intent="why",
        metadata={"k": "v", "n": 1, "f": 0.1, "nested": {"z": [1, 2.5, None, True]}},
        kind="fact",
        outcome="success",
        event_at="2026-10-01T00:00:00+00:00",
        valid_from=TS,
        valid_to="2027-01-01T00:00:00+00:00",
        supersedes_id="old-1",
        source_tier="web",
        schema_version=2,
    ),
)
case("v1-no-v2-keys", base(id="v1-a", layer="mid-term", tags=["old"], emotional={"intensity": 3, "valence": -1}))
case("v1-minimal", base(id="v1-min"))
case(
    "unknown-vocab",
    base(
        id="unk-1",
        kind="hypothesis",
        outcome="partial",
        source_tier="oracle",
        metadata={"keep": 1},
        schema_version=3,
    ),
)
case("unknown-kind-null", base(id="unk-null", kind=None, source_tier=None))
case(
    "unknown-preexisting-stash",
    base(id="unk-pre", kind="episode", metadata={"schema_v2_unknown": {"kind": "future-kind", "other": [1]}}),
)
case(
    "extra-fields-order",
    {
        "zzz": {"b": 1, "a": [0.5, 2]},
        "id": "extra-1",
        "title": "T",
        "content": "C",
        "created_at": TS,
        "updated_at": TS,
        "aaa": "after",
        "emotional": {"intensity": 1.0, "valence": 0.0, "future_emotion_key": "dropped"},
        "mmm": None,
    },
)
case(
    "float-formats",
    base(
        id="floats-1",
        emotional={"intensity": 5, "valence": 1e-05, "labels": []},
        metadata={
            "a": 1e5, "b": 1e16, "c": -0.0, "d": 1.0, "e": 0.1, "f": 1.5e-7, "g": 123456789012345678,
            "h": 18446744073709551615, "i": -9223372036854775808, "j": 2.5, "k": 1e22, "l": 123456.789,
            "m": 0.00012, "n": 5e-324, "o": 1.7976931348623157e308, "p": 100.0, "q": 1e15,
        },
    ),
)
case("rounding-half", base(id="round-1", emotional={"intensity": 0.25, "valence": -0.04, "labels": ["b", "a"]}))
case("rounding-half-2", base(id="round-2", emotional={"intensity": 0.35, "valence": 0.05}))
case("strip-title-id", base(id="  padded-id  ", title="  padded title 　"))
case("empty-containers", base(id="empty-1", metadata={"e": {}, "l": [], "s": ""}, tags=[], related_ids=[]))
case("explicit-null-temporal", base(id="null-t", event_at=None, valid_from=None, valid_to=None))
case("string-schema-keys-last", base(id="late-1", schema_version=2, kind="rule", outcome="lesson", source_tier="tool-output"))
case(
    "sealed",
    base(
        id="seal-1",
        title="Sealed",
        content="body",
        emotional={"intensity": 7.25, "valence": 0.55, "labels": ["joy", "trust"]},
    ),
)

INVALID = {
    "bad-layer": base(id="bad-1", layer="forever"),
    "bad-role": base(id="bad-2", role="root"),
    "empty-title": base(id="bad-3", title="   "),
    "intensity-range": base(id="bad-4", emotional={"intensity": 11}),
    "valence-range": base(id="bad-5", emotional={"valence": -2}),
    "metadata-list": base(id="bad-6", metadata=[1]),
    "empty-id": base(id="  "),
    "tags-not-list": base(id="bad-7", tags="x"),
    "missing-title": {"id": "bad-8", "content": "C"},
}


def dump(m: Memory) -> str:
    return json.dumps(m.model_dump(), indent=2, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()
    (HERE / "in").mkdir(exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)

    if args.corpus:
        for i, line in enumerate(Path(args.corpus).read_text().splitlines()):
            d = json.loads(line)
            d.setdefault("updated_at", d.get("created_at", TS))  # Python would stamp now()
            CASES[f"corpus-{i:02d}"] = d

    manifest: dict = {"cases": {}, "invalid": [], "tombstones": {}}
    for name, rec in CASES.items():
        raw = json.dumps(rec, indent=2)
        (HERE / "in" / f"{name}.json").write_text(raw)
        m = Memory.from_stored_json(raw)
        out = dump(m)
        (HERE / "out" / f"{name}.json").write_text(out)
        entry = {
            "integrity_hash": m.compute_integrity_hash(),
            "signature": m.emotional.signature(),
        }
        manifest["cases"][name] = entry
    # A sealed save: seal() then dump, so out/ carries a real integrity_hash.
    m = Memory.from_stored_json(json.dumps(CASES["sealed"]))
    m.seal()
    (HERE / "out" / "sealed-saved.json").write_text(dump(m))
    manifest["sealed_saved"] = {"source": "sealed", "integrity_hash": m.integrity_hash}

    for name, rec in INVALID.items():
        raw = json.dumps(rec)
        try:
            Memory.from_stored_json(raw)
        except Exception:
            (HERE / "in" / f"invalid-{name}.json").write_text(raw)
            manifest["invalid"].append(f"invalid-{name}")
        else:
            sys.exit(f"{name}: Python accepted a case expected to be invalid")

    for name, payload in {
        "with-agent": {"id": "tomb-1", "forgotten_at": "2026-11-07T09:30:00-0500", "reason": "forget", "agent": "lumina"},
        "no-agent": {"id": "tomb-2", "forgotten_at": "2026-11-07T14:30:00+0000", "reason": "dedupe"},
    }.items():
        manifest["tombstones"][name] = {"payload": payload, "bytes": json.dumps(payload)}

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(CASES)} cases, {len(INVALID)} invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
