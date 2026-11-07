#!/usr/bin/env python3
"""Generate the Python-written parity vectors for `snapshot` write-time behaviour (SKMEM-R-07C part 2).

Usage (reads Python skmemory, writes only under tests/snapshot_parity/):

    PYTHONPATH=<skmemory checkout> python tests/snapshot_parity/gen_snapshot_parity.py

Runs the real `MemoryStore.snapshot` against a throwaway SQLite primary in a temp dir (no vector,
no graph, no skseed, so its WAL lands in the temp dir too) over `CASES`, and records what Python did:
the derived `source_tier`, the inferred `valence`, the written `metadata` (key order included), the
dedup outcome (which earlier case's id came back) and whether Python decomposed the content. The
Rust test (`tests/snapshot_parity.rs`) replays the same calls and must reproduce every value.
Synthetic data only. Regenerate, never hand-edit.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="skmem-snapgen-"))
os.environ["SKCAPSTONE_HOME"] = str(tmp / "home")
os.environ["SKMEMORY_HOME"] = str(tmp / "home")
os.environ.pop("SKMEMORY_READER", None)
os.environ["SKMEMORY_SKSEED_AUTO_VALIDATE"] = "0"

from skmemory.backends.sqlite_backend import SQLiteBackend  # noqa: E402
from skmemory.models import EmotionalSnapshot, MemoryLayer  # noqa: E402
from skmemory.store import MemoryStore  # noqa: E402

HERE = Path(__file__).resolve().parent
LONG = ("A long paragraph about the river delta and the amberfox. " * 30).strip()
assert len(LONG) >= 1200

CASES = [
    {"name": "plain-manual", "title": "plain", "content": "hello world one"},
    {"name": "dup-same-content", "title": "other title", "content": "hello world one"},
    {"name": "src-web", "title": "t web", "content": "web content a", "source": "web-fetch"},
    {"name": "src-http-upper", "title": "t http", "content": "web content b", "source": "HTTP-client"},
    {"name": "src-browser", "title": "t br", "content": "web content c", "source": "browser"},
    {"name": "src-generated", "title": "t gen", "content": "gen content a", "source": "skwhisper"},
    {"name": "src-generated-mcp", "title": "t gen2", "content": "gen content b", "source": "mcp:memory_save_session"},
    {"name": "src-tool", "title": "t tool", "content": "tool content a", "source": "tool"},
    {"name": "src-taskpack", "title": "t tp", "content": "tool content b", "source": "task-pack"},
    {"name": "src-user-telegram", "title": "t tg", "content": "user content a", "source": "telegram"},
    {"name": "src-import", "title": "t imp", "content": "user content b", "source": "import-notion-2026"},
    {"name": "src-unknown", "title": "t unk", "content": "unk content", "source": "zzz-unknown"},
    {"name": "src-padded-case", "title": "t pad", "content": "pad content", "source": "  Manual "},
    {"name": "explicit-tier", "title": "t et", "content": "explicit tier content", "source": "web-fetch", "source_tier": "user-stated"},
    {"name": "val-positive", "title": "v1", "content": "val content 1", "emotional": {"labels": ["joy", "trust"]}},
    {"name": "val-one-positive", "title": "v1b", "content": "val content 1b", "emotional": {"labels": ["Gratitude"]}},
    {"name": "val-many-positive", "title": "v1c", "content": "val content 1c", "emotional": {"labels": ["joy", "love", "hope", "happy", "trust"]}},
    {"name": "val-negative", "title": "v2", "content": "val content 2", "emotional": {"labels": ["fear", "anger", "grief"]}},
    {"name": "val-tie", "title": "v3", "content": "val content 3", "emotional": {"labels": ["joy", "fear"]}},
    {"name": "val-explicit", "title": "v4", "content": "val content 4", "emotional": {"labels": ["joy"], "valence": -0.25}},
    {"name": "val-unknown-label", "title": "v5", "content": "val content 5", "emotional": {"labels": ["serene"]}},
    {"name": "val-intensity", "title": "v6", "content": "val content 6", "emotional": {"labels": ["sadness"], "intensity": 7.5}},
    {"name": "auth-case", "title": "Smith v. Jones opinion", "content": "auth content 1"},
    {"name": "auth-statute", "title": "26 USC 501 summary", "content": "auth content 2"},
    {"name": "auth-rule", "title": "Federal rules", "content": "auth content 3", "tags": ["usc", "civil procedure"]},
    {"name": "auth-form", "title": "IRS Form 1040 notes", "content": "auth content 4"},
    {"name": "auth-template", "title": "meeting template", "content": "auth content 5"},
    {"name": "auth-secondary", "title": "Black's Law handbook", "content": "auth content 6"},
    {"name": "auth-source-ref", "title": "plain title", "content": "auth content 7", "source_ref": "reference/treatise.pdf"},
    {"name": "auth-explicit", "title": "plain title 2", "content": "auth content 8", "metadata": {"authority_tier": " RULE "}},
    {"name": "auth-explicit-bad", "title": "plain title 3", "content": "auth content 9", "metadata": {"authority_tier": "bogus"}},
    {"name": "auth-metadata-keep", "title": "plain title 4", "content": "auth content 10", "metadata": {"b": 1, "a": {"x": [1, 2]}, "file_path": "generated/x.md"}},
    {"name": "layer-long", "title": "t lay", "content": "layer content", "layer": "long-term", "tags": ["x", "y"]},
    {"name": "decompose-long", "title": "t long", "content": LONG},
    {"name": "decompose-flag", "title": "t flag", "content": "short but flagged", "decompose": True},
]


def kwargs(c):
    kw = {}
    for k in ("source", "tags", "source_ref", "metadata", "decompose"):
        if k in c:
            kw[k] = c[k]
    if "layer" in c:
        kw["layer"] = MemoryLayer(c["layer"])
    if "source_tier" in c:
        kw["source_tier"] = c["source_tier"]
    if "emotional" in c:
        kw["emotional"] = EmotionalSnapshot(**c["emotional"])
    return kw


try:
    store = MemoryStore(primary=SQLiteBackend(str(tmp / "mem")), skseed_auto_validate=False)
    ids = {}
    out = []
    for c in CASES:
        m = store.snapshot(c["title"], c["content"], **kwargs(c))
        first = next((n for n, i in ids.items() if i == m.id), None)
        ids.setdefault(c["name"], m.id)
        out.append(
            {
                "case": c,
                "same_as": first,
                "decomposed": bool(m.metadata.get("decomposition")) or bool(m.related_ids),
                "source_tier": m.source_tier,
                "valence": m.emotional.valence,
                "intensity": m.emotional.intensity,
                "labels": list(m.emotional.labels),
                "metadata_keys": list(m.metadata),
                "metadata": m.metadata,
                "layer": m.layer.value,
                "tags": list(m.tags),
                "source": m.source,
                "title": m.title,
            }
        )
    (HERE / "expected.json").write_text(json.dumps(out, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases to {HERE / 'expected.json'}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
