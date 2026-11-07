#!/usr/bin/env python3
"""Python side of the pgvector arm parity test (SKMEM-R-04).

Usage (needs a THROWAWAY skmem-pg with pgvector; never a real node):

    SKMEMORY_PG_DSN=<throwaway dsn> PYTHONPATH=<skmemory checkout> \
        python tests/pg_parity/gen_pg_parity.py tests/pg_parity

Builds a corpus, seeds it into ``public.memories`` under agents prefixed
``skrs-r04-``, runs a query matrix through the REAL Python
``PGVectorBackend.search_semantic`` (with an injected deterministic embedder
that the Rust test reproduces byte for byte), and writes:

* ``corpus.json``   rows to seed (memory_json as Python's model_dump_json wrote it)
* ``expected.json`` for each case, the ordered ids Python returned (or an error)

It deletes its rows again. Regenerate with this script, never by hand.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from skmemory.backends.pgvector_backend import PGVectorBackend
from skmemory.embed_text import embed_text
from skmemory.models import Memory, MemoryLayer
from skmemory.validity import normalize_as_of

OUT = Path(sys.argv[1])
DSN = os.environ["SKMEMORY_PG_DSN"]
M64 = (1 << 64) - 1
DIM = 1024


def fnv1a64(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for c in b:
        h = ((h ^ c) * 0x100000001B3) & M64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def embed_fn(text: str) -> list[float]:
    acc = [0] * DIM
    for tok in text.lower().split():
        h = fnv1a64(tok.encode())
        for i in range(DIM):
            acc[i] += ((splitmix64((h + i) & M64) >> 33) % 1001) - 500
    return [a / 64.0 for a in acc]


WORDS = (
    "river stone amber falcon garden copper lantern meadow harbor quartz "
    "violet thunder maple ember glacier orchid saffron timber velvet willow "
    "zephyr compass anchor beacon cinder dune echo fjord grove haze iris"
).split()
LAYERS = ["short-term", "mid-term", "long-term"]
TAGS = ["alpha", "beta", "gamma", "delta"]
SOURCES = ["seed", "chat", "import"]
CONTEXTS = ["public", "team", "private", ""]

rng = random.Random(404)
rows = []


def make(idx: int, agent: str, **over) -> dict:
    words = [rng.choice(WORDS) for _ in range(rng.randint(6, 14))]
    month = 1 + (idx % 10)
    created = f"2026-{month:02d}-{1 + idx % 27:02d}T{idx % 24:02d}:30:00+00:00"
    f = dict(
        id=f"skrs-r04-{idx:03d}",
        title=f"{words[0]} {words[1]} note",
        content=" ".join(words),
        summary=" ".join(words[:3]) if idx % 3 == 0 else "",
        layer=LAYERS[idx % 3],
        tags=[t for j, t in enumerate(TAGS) if (idx >> j) & 1],
        source=SOURCES[idx % 3],
        context_tag=CONTEXTS[idx % 4],
        created_at=created,
        valid_from=None,
        valid_to=None,
    )
    f.update(over)
    return dict(agent=agent, **f)


for i in range(48):
    rows.append(make(i, "skrs-r04-a"))
for i in range(48, 60):
    rows.append(make(i, "skrs-r04-b"))
# validity windows, including exact boundaries
rows[3]["valid_from"] = "2026-03-01T00:00:00+00:00"
rows[3]["valid_to"] = "2026-06-01T00:00:00+00:00"
rows[4]["valid_to"] = "2026-02-01T00:00:00+00:00"
rows[5]["valid_from"] = "2026-09-01T00:00:00+00:00"
rows[6]["valid_from"] = "2026-06-01T00:00:00+00:00"
rows[7]["valid_to"] = "2026-06-01T02:00:00+02:00"
rows[8]["valid_from"] = "2026-05-31T22:00:00-02:00"
for k in (10, 11, 12):
    rows[k]["valid_to"] = f"2026-0{k - 5}-15T12:00:00+00:00"
OPEN_ROW = 20  # v1 row: no v2 columns, no valid_* in its json

bodies = []
for n, r in enumerate(rows):
    m = Memory(
        id=r["id"],
        title=r["title"],
        content=r["content"],
        summary=r["summary"],
        layer=MemoryLayer(r["layer"]),
        tags=r["tags"],
        source=r["source"],
        context_tag=r["context_tag"],
        created_at=r["created_at"],
        updated_at=r["created_at"],
        valid_from=r["valid_from"],
        valid_to=r["valid_to"],
    )
    body = dict(r)
    body["memory_json"] = json.loads(m.model_dump_json())
    body["v2"] = n != OPEN_ROW
    body["updated_at"] = m.updated_at
    body["role"] = str(m.role.value)
    body["kind"] = m.kind
    body["outcome"] = m.outcome
    body["event_at"] = m.event_at
    body["supersedes_id"] = m.supersedes_id
    body["source_tier"] = m.source_tier
    body["schema_version"] = m.schema_version
    body["embed_text"] = embed_text(m.title, m.content, m.summary)
    body["has_embedding"] = True
    bodies.append(body)

# a row with no embedding (must never be returned) and one with a broken
# memory_json (missing title: skipped, counted, not fatal)
nb = make(60, "skrs-r04-a", title="nullvec", content="river stone", summary="")
nb_m = Memory(id=nb["id"], title=nb["title"], content=nb["content"], created_at=nb["created_at"], updated_at=nb["created_at"])
bodies.append(
    dict(
        nb,
        memory_json=json.loads(nb_m.model_dump_json()),
        v2=True,
        updated_at=nb_m.updated_at,
        role="assistant",
        kind=nb_m.kind,
        outcome=None,
        event_at=None,
        supersedes_id=None,
        source_tier=nb_m.source_tier,
        schema_version=nb_m.schema_version,
        embed_text=embed_text(nb["title"], nb["content"], ""),
        has_embedding=False,
    )
)
bad = make(61, "skrs-r04-a", title="brokenjson", content="river stone amber", summary="")
bodies.append(
    dict(
        bad,
        memory_json={"id": bad["id"], "content": bad["content"]},
        v2=True,
        updated_at=bad["created_at"],
        role="assistant",
        kind=None,
        outcome=None,
        event_at=None,
        supersedes_id=None,
        source_tier=None,
        schema_version=None,
        embed_text=embed_text(bad["title"], bad["content"], ""),
        has_embedding=True,
    )
)

A, B = "skrs-r04-a", "skrs-r04-b"
Q = {
    "river": "river stone amber",
    "falcon": "falcon harbor lantern copper",
    "grove": "grove haze iris echo",
    "mixed": "willow velvet anchor beacon dune",
}


def case(name, q, agent=A, limit=10, **f):
    return dict(name=name, query=Q[q] if q in Q else q, agent=agent, limit=limit, filters=f)


cases = [
    case("plain", "river"),
    case("plain_limit1", "falcon", limit=1),
    case("plain_limit100", "grove", limit=100),
    case("agent_b", "mixed", agent=B),
    case("layer_short", "river", layer="short-term"),
    case("layer_long", "falcon", layer="long-term"),
    case("tags_one", "river", tags=["alpha"]),
    case("tags_and", "mixed", tags=["alpha", "gamma"]),
    case("tags_and3", "grove", tags=["alpha", "beta", "gamma"]),
    case("tags_none_match", "river", tags=["nosuchtag"]),
    case("tags_empty_is_no_filter", "river", tags=[]),
    case("ctx_one_public", "river", context_tag="public"),
    case("ctx_one_empty_string", "mixed", context_tag=""),
    case("ctx_any", "falcon", context_tag=["public", "team"]),
    case("ctx_any_single", "grove", context_tag=["private"]),
    case("ctx_empty_list_fails_closed", "river", context_tag=[]),
    case("ctx_unknown", "river", context_tag="nope"),
    case("source_chat", "mixed", source="chat"),
    case("source_empty_is_no_filter", "mixed", source=""),
    case("layer_tags_ctx", "river", layer="mid-term", tags=["beta"], context_tag=["public", "private", "team"]),
    case("everything", "mixed", layer="long-term", tags=["gamma"], context_tag="public", source="import"),
    case("asof_date_only", "river", as_of="2026-04-10"),
    case("asof_z", "falcon", as_of="2026-04-10T00:00:00Z"),
    case("asof_naive", "grove", as_of="2026-04-10T05:00:00"),
    case("asof_offset", "mixed", as_of="2026-04-10T05:00:00+05:00"),
    case("asof_valid_to_boundary_excluded", "river", as_of="2026-06-01T00:00:00+00:00", limit=100),
    case("asof_valid_from_boundary_included", "river", as_of="2026-06-01", limit=100),
    case("asof_offset_boundaries", "river", as_of="2026-06-01T00:00:00Z", limit=100),
    case("asof_before_everything", "river", as_of="2025-01-01", limit=100),
    case("asof_far_future", "river", as_of="2030-01-01", limit=100),
    case("asof_empty_is_no_filter", "river", as_of="", limit=100),
    case("asof_with_layer", "falcon", as_of="2026-05-15", layer="short-term", limit=100),
    case("asof_bad", "river", as_of="not-a-date"),
    case("asof_bad_now", "river", as_of="now"),
    case("asof_bad_yesterday", "river", as_of="yesterday"),
    case("asof_bad_today", "river", as_of="today"),
    case("asof_bad_infinity", "river", as_of="infinity"),
    case("asof_bad_epoch", "river", as_of="epoch"),
    case("asof_bad_month", "river", as_of="2026-13-45"),
    case("asof_bad_feb30", "river", as_of="2026-02-30"),
    case("asof_bad_hour24", "river", as_of="2026-04-10T24:00:00"),
    case("asof_whitespace_is_no_filter", "river", as_of="   ", limit=100),
    case("asof_compact_date", "falcon", as_of="20260410"),
    case("asof_space_separator_fraction", "grove", as_of="2026-04-10 05:00:00.250+02:00"),
    case("asof_agent_b", "mixed", agent=B, as_of="2026-12-31", limit=100),
    case("unmatched_filters_empty", "river", layer="short-term", tags=["alpha", "beta", "gamma", "delta"], source="import"),
]

with psycopg.connect(DSN, autocommit=True) as conn:
    register_vector(conn)
    conn.execute("DELETE FROM memories WHERE id LIKE 'skrs-r04-%'")
    for b in bodies:
        vec = embed_fn(b["embed_text"]) if b["has_embedding"] else None
        cols = (
            "id, agent, layer, role, title, content, summary, tags, source, created_at, "
            "updated_at, memory_json, embedding"
        )
        vals = [b["id"], b["agent"], b["layer"], b["role"], b["title"], b["content"], b["summary"] or "", b["tags"],
                b["source"], b["created_at"], b["updated_at"], json.dumps(b["memory_json"]), vec]
        ph = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s::jsonb,%s"
        if b["v2"]:
            cols += ", kind, outcome, event_at, valid_from, valid_to, supersedes_id, source_tier, schema_version"
            ph += ",%s,%s,%s::timestamptz,%s::timestamptz,%s::timestamptz,%s,%s,%s"
            vals += [b["kind"], b["outcome"], b["event_at"], b["valid_from"], b["valid_to"],
                     b["supersedes_id"], b["source_tier"], b["schema_version"]]
        conn.execute(f"INSERT INTO memories ({cols}) VALUES ({ph})", vals)

    be = PGVectorBackend(dsn=DSN, embed_fn=embed_fn, agent=os.environ.get("SKAGENT") or "lumina")
    out = []
    for c in cases:
        be.agent = c["agent"]
        f = dict(c["filters"])
        kw = {}
        if "layer" in f:
            kw["layer"] = MemoryLayer(f["layer"])
        for k in ("tags", "context_tag", "source"):
            if k in f:
                kw[k] = f[k]
        entry = dict(c)
        try:
            if "as_of" in f:
                kw["as_of"] = normalize_as_of(f["as_of"])
            ms = be.search_semantic(c["query"], c["limit"], **kw)
            entry["expect_ids"] = [m.id for m in ms]
            q = embed_fn(c["query"])
            drows = dict(
                conn.execute(
                    "SELECT id, embedding <=> %s::vector FROM memories WHERE id = ANY(%s) AND agent = %s",
                    (q, entry["expect_ids"], c["agent"]),
                ).fetchall()
            )
            entry["expect_distances"] = [round(float(drows[i]), 6) for i in entry["expect_ids"]]
        except Exception as e:  # noqa: BLE001
            entry["expect_error"] = type(e).__name__
        out.append(entry)
    conn.execute("DELETE FROM memories WHERE id LIKE 'skrs-r04-%'")

(OUT / "corpus.json").write_text(json.dumps(bodies, indent=1) + "\n", encoding="utf-8")
NORM_INPUTS = [
    "2026-04-10", "20260410", "2026-04-10T05:00:00", "2026-04-10T05:00:00Z", "2026-04-10t05:00:00z",
    "2026-04-10 05:00:00.5", "2026-04-10T05:00:00.123456789+05:30", "2026-04-10T05:00", "2026-04-10T05",
    "2026-04-10T0500", "2026-04-10T050007", "2026-04-10T05:00:00-02:00", "2026-04-10T05:00:00+0530",
    "2026-04-10T05:00:00+05", "  2026-04-10  ", "2024-02-29", "2026-12-31T23:59:59.999999-23:59",
    "0001-01-01", "9999-12-31T23:59:59Z", "9999-12-31T23:59:59-01:00", "0001-01-01T00:00:00+01:00",
    "", "   ", "now", "yesterday", "today", "tomorrow", "infinity", "-infinity", "epoch", "garbage",
    "2026-13-01", "2026-00-10", "2026-02-30", "2025-02-29", "2026-04-31", "2026-04-10T24:00",
    "2026-04-10T05:60", "2026-04-10T05:00:60", "2026-04-10T05:00:00+24:00", "2026-04-10T05:00:00+05:60",
    "2026-04-10T", "2026-04-1", "26-04-10", "2026/04/10", "2026-04-10T05:00:00 +05:00", "2026-04-10Z",
    "1 day ago", "12/31/2026", "2026-04-10T05:00:00.", "2026-04-10T5:00",
]
norm = []
for v in NORM_INPUTS:
    try:
        norm.append({"input": v, "out": normalize_as_of(v)})
    except (ValueError, OverflowError):
        norm.append({"input": v, "error": True})
(OUT / "asof_norm.json").write_text(json.dumps(norm, indent=1) + "\n", encoding="utf-8")
(OUT / "expected.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(len(bodies), "rows,", len(out), "cases,", sum(1 for e in out if "expect_error" in e), "errors,",
      sum(1 for e in out if e.get("expect_ids") == []), "empty")
