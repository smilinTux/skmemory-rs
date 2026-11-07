#!/usr/bin/env python3
"""Python side of the SQLite index parity test (SKMEM-R-03).

Usage (reads Python skmemory; writes only under <work-dir> and <out.json>):

    PYTHONPATH=<skmemory checkout> python tests/golden/index_parity.py \
        tests/golden/out <work-dir> tests/golden/index/py_index.json

Builds an index over the golden corpus with the REAL Python SQLiteBackend and
records everything Rust must reproduce: every ``memories`` column (file_path
relative to the tree), the ``fts_meta`` sidecar, the schema text in
``sqlite_master``, ordered bm25 results for a query set, and the usage state
after a fixed counter scenario and a reindex. Memories are indexed in the
order Rust walks them (layer order, sorted file names) so FTS rowids match.
Regenerate with this script, never by hand.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

from skmemory import fts_local
from skmemory.backends.sqlite_backend import SQLiteBackend
from skmemory.models import Memory, MemoryLayer

src, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tree = work / "tree"
shutil.rmtree(tree, ignore_errors=True)
for layer in MemoryLayer:
    (tree / layer.value).mkdir(parents=True)

loaded = []
for f in sorted(src.glob("*.json")):
    try:
        m = Memory.from_stored(json.loads(f.read_text(encoding="utf-8")))
    except Exception:
        continue
    dest = tree / m.layer.value / f"{m.id}.json"
    shutil.copyfile(f, dest)
    loaded.append(m.id)

be = SQLiteBackend(base_path=str(tree))
assert be._fts_ok, "FTS5 unavailable in this Python sqlite"
for layer in MemoryLayer:
    for p in sorted((tree / layer.value).glob("*.json")):
        be._index_memory(Memory.from_stored(json.loads(p.read_text(encoding="utf-8"))), p)

conn = sqlite3.connect(str(tree / "index.db"))
conn.row_factory = sqlite3.Row


def rows():
    out = []
    for r in conn.execute("SELECT * FROM memories ORDER BY id"):
        d = dict(r)
        d["file_path"] = str(Path(d["file_path"]).relative_to(tree))
        out.append(d)
    return out


def meta():
    return [dict(r) for r in conn.execute("SELECT rowid, id, content_hash, context_tag FROM fts_meta ORDER BY rowid")]


def schema():
    return [
        dict(r)
        for r in conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' AND name NOT LIKE 'memory_fts_%' ORDER BY type, name"
        )
    ]


mems = [Memory.from_stored(json.loads(p.read_text(encoding="utf-8"))) for l in MemoryLayer for p in sorted((tree / l.value).glob("*.json"))]
words = []
for m in mems:
    words += [w for w in (m.title or "").split()[:2]]
queries = [{"q": w} for w in dict.fromkeys(words)][:40]
queries += [
    {"q": "memory continuity"},
    {"q": "nosuchwordanywhere"},
    {"q": "prb-6f069c5e AND NEAR(x)"},
    {"q": '"quoted" (paren) col:umn'},
    {"q": "   "},
    {"q": "love", "layer": "long-term"},
    {"q": "love", "source": "manual"},
    {"q": "the", "tags": ["cloud9"]},
    {"q": "love", "context_tag": []},
    {"q": "love", "context_tag": [""]},
    {"q": "love", "as_of": "2026-10-02T12:00:00+00:00"},
    {"q": "love", "as_of": "2001-01-01T00:00:00+00:00"},
]
res = []
for q in queries:
    r = fts_local.search_rows(
        conn, q["q"], 10, layer=q.get("layer"), tags=q.get("tags"), source=q.get("source"),
        context_tag=q.get("context_tag"), as_of=q.get("as_of"),
    )
    res.append({**q, "ids": [x["id"] for x in r]})

rows0, meta0, schema0 = rows(), meta(), schema()
# Counter scenario. "now" far in the future so the 1h debounce skips it.
ids = sorted(set(loaded))
FUT = "2999-01-01T00:00:00+00:00"
b1 = be.bump_access(ids[:3], now=FUT)
b2 = be.bump_access(ids[:3], now=FUT)
b3 = be.bump_access(ids[:2], debounce_s=3600)  # last_accessed is in the future: skipped
b4 = be.bump_access(ids[3:5], now="2026-01-01T00:00:00+00:00", debounce_s=3600)
# A legacy 0.5 row (no accesses) whose recompute differs, then a re-save UPSERT.
legacy = next(r["id"] for r in rows() if r["access_count"] == 0 and abs(r["importance"] - 0.5) > 1e-9)
conn.execute("UPDATE memories SET importance = 0.5 WHERE id = ?", (legacy,))
conn.commit()
usage = be.get_usage(ids[:6])
after_bump = rows()
n = be.reindex(force=True)
after_reindex = rows()
legacy2 = next(r["id"] for r in after_reindex if r["access_count"] == 0 and abs(r["importance"] - 0.5) > 1e-9 and r["id"] != legacy)
conn.execute("UPDATE memories SET importance = 0.5 WHERE id = ?", (legacy2,))
conn.commit()
imp = be.backfill_importance()
after_backfill = rows()
fts_cov = fts_local.coverage(conn)
conn.close()
be.close()

doc = {
    "ids": ids,
    "legacy_id": legacy,
    "legacy2_id": legacy2,
    "rows": rows0,
    "fts_meta": meta0,
    "fts_coverage": fts_cov,
    "after_backfill": after_backfill,
    "bumps": [b1, b2, b3, b4],
    "usage": {k: v for k, v in usage.items()},
    "reindex_count": n,
    "after_bump": after_bump,
    "after_reindex": after_reindex,
    "backfill_importance": imp,
    "schema": schema0,
    "queries": res,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(len(ids), "memories,", len(res), "queries")
