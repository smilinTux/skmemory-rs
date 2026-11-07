#!/usr/bin/env python3
"""Python side of the lexical and graph arm parity tests (SKMEM-R-05).

Usage (needs a THROWAWAY skmem-pg with pgvector, pg_search and AGE; never a
real node), with ``PYTHONPATH`` at a checkout of skmemory ``origin/main``:

    SKMEMORY_PG_DSN=<throwaway dsn> PYTHONPATH=<skmemory checkout> \
        python tests/pg_parity/gen_r05_parity.py tests/pg_parity

Seeds a corpus into ``public.memories`` under agents prefixed ``skrs-r05-``
and into a scratch AGE graph, runs a case matrix through the REAL Python
``PGVectorBackend.search_lexical`` and ``AGEGraphBackend`` read methods, and
writes ids AND scores:

* ``r05_lex_corpus.json`` / ``r05_lex_expected.json``
* ``r05_graph_corpus.json`` / ``r05_graph_expected.json``

Everything it created is deleted again. The AGE backend is built with
``__new__`` and explicit fields so no agent profile (no real store) is read.
Regenerate with this script, never by hand.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import psycopg

from skmemory.backends.age_backend import AGEGraphBackend
from skmemory.backends.pgvector_backend import PGVectorBackend
from skmemory.models import Memory, MemoryLayer
from datetime import datetime, timezone

from skmemory.validity import normalize_as_of, parse_ts

OUT = Path(sys.argv[1])
DSN = os.environ["SKMEMORY_PG_DSN"]
GRAPH = "skrs_r05_knowledge"
A, B = "skrs-r05-a", "skrs-r05-b"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


# --------------------------------------------------------------------------
# Lexical corpus
# --------------------------------------------------------------------------
WORDS = (
    "river stone amber falcon garden copper lantern meadow harbor quartz "
    "violet thunder maple ember glacier orchid saffron timber velvet willow "
    "zephyr compass anchor beacon cinder dune echo fjord grove haze iris"
).split()
FOREIGN = [
    "Zürich café naïve résumé",
    "東京 大阪 京都",
    "straße Ærø Ångström",
    "señor niño mañana",
    "Привет мир river",
    "emoji 🌊 wave harbor",
]
LAYERS = ["short-term", "mid-term", "long-term"]
TAGS = ["alpha", "beta", "gamma", "delta"]
SOURCES = ["seed", "chat", "import"]
rng = random.Random(505)
rows: list[dict] = []


def make(idx: int, agent: str, **over) -> dict:
    words = [rng.choice(WORDS) for _ in range(rng.randint(5, 16))]
    month = 1 + (idx % 10)
    created = f"2026-{month:02d}-{1 + idx % 27:02d}T{idx % 24:02d}:30:00+00:00"
    f = dict(
        id=f"skrs-r05-{idx:03d}",
        title=f"{words[0]} {words[1]} note",
        content=" ".join(words),
        summary=" ".join(words[:3]) if idx % 3 == 0 else "",
        layer=LAYERS[idx % 3],
        tags=[t for j, t in enumerate(TAGS) if (idx >> j) & 1],
        source=SOURCES[idx % 3],
        context_tag="public",
        created_at=created,
        valid_from=None,
        valid_to=None,
    )
    f.update(over)
    return dict(agent=agent, **f)


for i in range(48):
    rows.append(make(i, A))
for i in range(48, 60):
    rows.append(make(i, B))
for k, text in enumerate(FOREIGN):
    r = make(
        60 + k,
        A,
        title=text.split()[0] + " " + text.split()[-1],
        content=f"{text} {rng.choice(WORDS)}",
    )
    rows.append(r)
# a terse identifier record plus noisy neighbours (the BM25 rationale)
rows.append(make(70, A, title="prb-6f069c5e", content="remediation", summary=""))
rows.append(
    make(
        71,
        A,
        title="plan",
        content="per node postgres rebuild remediation plan for the swarm",
        summary="",
    )
)
rows.append(
    make(
        72,
        A,
        title="Zürich plan",
        content="Zürich café plan for naïve river harbor",
        summary="harbor",
    )
)
# validity windows (boundaries and mixed offsets)
rows[3]["valid_from"] = "2026-03-01T00:00:00+00:00"
rows[3]["valid_to"] = "2026-06-01T00:00:00+00:00"
rows[4]["valid_to"] = "2026-02-01T00:00:00+00:00"
rows[5]["valid_from"] = "2026-09-01T00:00:00+00:00"
rows[6]["valid_from"] = "2026-06-01T00:00:00+00:00"
rows[7]["valid_to"] = "2026-06-01T02:00:00+02:00"
rows[8]["valid_from"] = "2026-05-31T22:00:00-02:00"
for k in (10, 11, 12):
    rows[k]["valid_to"] = f"2026-0{k - 5}-15T12:00:00+00:00"
# the top scorers for "zürich café" closed early: forces the BM25 overfetch
for k in (60, 68):  # the two Zürich rows
    rows[k]["valid_to"] = "2026-01-02T00:00:00+00:00"
OPEN_ROW = 20

bodies: list[dict] = []
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
    bodies.append(body)
# a row whose memory_json cannot parse (skipped and counted) that matches "river"
bad = make(80, A, title="brokenjson", content="river river river stone", summary="")
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
    )
)


def lcase(name, q, agent=A, limit=10, **f):
    return dict(name=name, query=q, agent=agent, limit=limit, filters=f)


LEX_QUERIES = {
    "river": "river stone amber",
    "falcon": "falcon harbor lantern",
    "single": "river",
    "zurich": "zürich café",
    "zurich_upper": "ZÜRICH",
    "tokyo": "東京",
    "strasse": "straße",
    "cyrillic": "Привет",
    "ident": "prb-6f069c5e",
    "plan": "per node postgres rebuild remediation plan prb-6f069c5e",
    "stop": "the and of",
    "none": "qqqzzzxxx",
    "empty": "",
    "space": "   ",
    "sql": "'; DROP TABLE memories; --",
    "tsops": "river & !stone | falcon:* <-> harbor",
    "like": "100%_\\ river",
    "dollar": "$$ river $1 $2",
    "long": " ".join(WORDS * 8),
}
lex_cases = []
for qn, q in LEX_QUERIES.items():
    lex_cases.append(lcase(f"q_{qn}", q))
lex_cases += [
    lcase("limit1", LEX_QUERIES["river"], limit=1),
    lcase("limit100", LEX_QUERIES["single"], limit=100),
    lcase("agent_b", LEX_QUERIES["river"], agent=B, limit=20),
    lcase("layer", LEX_QUERIES["river"], layer="short-term", limit=50),
    lcase("tags_and", LEX_QUERIES["river"], tags=["alpha", "gamma"], limit=50),
    lcase("tags_empty", LEX_QUERIES["river"], tags=[], limit=50),
    lcase("tags_none", LEX_QUERIES["river"], tags=["nosuch"]),
    lcase("source", LEX_QUERIES["falcon"], source="chat", limit=50),
    lcase("source_empty", LEX_QUERIES["falcon"], source="", limit=50),
    lcase(
        "all_filters",
        LEX_QUERIES["river"],
        layer="mid-term",
        tags=["beta"],
        source="chat",
        limit=50,
    ),
    lcase("asof_date", LEX_QUERIES["single"], as_of="2026-04-10", limit=100),
    lcase("asof_date_limit3", LEX_QUERIES["single"], as_of="2026-04-10", limit=3),
    lcase(
        "asof_boundary_closed",
        LEX_QUERIES["single"],
        as_of="2026-06-01T00:00:00+00:00",
        limit=100,
    ),
    lcase(
        "asof_boundary_offset",
        LEX_QUERIES["single"],
        as_of="2026-06-01T00:00:00Z",
        limit=100,
    ),
    lcase("asof_naive", LEX_QUERIES["river"], as_of="2026-04-10T05:00:00", limit=100),
    lcase(
        "asof_offset",
        LEX_QUERIES["river"],
        as_of="2026-04-10T05:00:00+05:00",
        limit=100,
    ),
    lcase("asof_before_all", LEX_QUERIES["single"], as_of="2025-01-01", limit=100),
    lcase("asof_future", LEX_QUERIES["single"], as_of="2030-01-01", limit=100),
    lcase("asof_empty", LEX_QUERIES["single"], as_of="", limit=100),
    lcase("asof_unicode_query", LEX_QUERIES["zurich"], as_of="2026-03-01", limit=5),
    lcase("asof_overfetch", LEX_QUERIES["zurich"], as_of="2026-12-01", limit=2),
    lcase(
        "asof_layer",
        LEX_QUERIES["falcon"],
        as_of="2026-05-15",
        layer="short-term",
        limit=100,
    ),
    lcase("asof_bad", LEX_QUERIES["single"], as_of="not-a-date"),
    lcase("asof_now", LEX_QUERIES["single"], as_of="now"),
    lcase("asof_feb30", LEX_QUERIES["single"], as_of="2026-02-30"),
]


def where_for(c) -> tuple[str, list]:
    f = c["filters"]
    clauses, params = ["agent=%s"], [c["agent"]]
    if f.get("layer"):
        clauses.append("layer=%s")
        params.append(f["layer"])
    if f.get("tags"):
        clauses.append("tags @> %s")
        params.append(list(f["tags"]))
    if f.get("source"):
        clauses.append("source=%s")
        params.append(f["source"])
    return " AND ".join(clauses), params


def run_lexical(conn) -> list[dict]:
    cap = _Capture()
    logging.getLogger("skmemory.backends.pgvector_backend").addHandler(cap)
    logging.getLogger("skmemory.backends.pgvector_backend").setLevel(logging.WARNING)
    # The constructor resolves an agent profile (a registry read); patch that one
    # call so no real agent home is consulted. Everything else is the real class.
    import skmemory.agents as _agents
    from types import SimpleNamespace

    real = _agents.require_memory_profile
    _agents.require_memory_profile = lambda agent=None, environment=None: (
        SimpleNamespace(profile_id=agent)
    )
    try:
        be = PGVectorBackend(dsn=DSN, embed_fn=lambda t: [], agent=A)
    finally:
        _agents.require_memory_profile = real
    out = []
    for mode in ("bm25", "tsrank"):
        for c in lex_cases:
            be.agent = c["agent"]
            be._bm25_available = None if mode == "bm25" else False
            f = dict(c["filters"])
            kw = {}
            if "layer" in f:
                kw["layer"] = MemoryLayer(f["layer"])
            for k in ("tags", "source"):
                if k in f:
                    kw[k] = f[k]
            entry = dict(c, mode=mode)
            cap.msgs.clear()
            try:
                if "as_of" in f:
                    kw["as_of"] = normalize_as_of(f["as_of"])
                ms = be.search_lexical(c["query"], c["limit"], **kw)
                ids = [m.id for m in ms]
                entry["bm25_failed"] = any(
                    "BM25 lexical search failed" in m for m in cap.msgs
                )
                entry["expect_ids"] = ids
                where, params = where_for(c)
                use_bm25 = (
                    mode == "bm25" and be._has_bm25_index() and not entry["bm25_failed"]
                )
                cur = conn.cursor()
                if use_bm25:
                    cur.execute(
                        f"SELECT id, paradedb.score(id)::float8 FROM memories WHERE {where} "
                        "AND id @@@ paradedb.boolean(should => ARRAY[paradedb.match('content', %s), "
                        "paradedb.match('title', %s)]) LIMIT 100000",
                        params + [c["query"], c["query"]],
                    )
                else:
                    cur.execute(
                        f"SELECT id, ts_rank(tsv, plainto_tsquery('english', %s))::float8 FROM memories "
                        f"WHERE {where} AND tsv @@ plainto_tsquery('english', %s)",
                        [c["query"]] + params + [c["query"]],
                    )
                scores = dict(cur.fetchall())
                entry["expect_scores"] = [float(scores[i]) for i in ids]
                entry["expect_path"] = "bm25" if use_bm25 else "tsrank"
            except Exception as e:  # noqa: BLE001
                entry["expect_error"] = type(e).__name__
            out.append(entry)
    logging.getLogger("skmemory.backends.pgvector_backend").removeHandler(cap)
    return out


# --------------------------------------------------------------------------
# Graph corpus
# --------------------------------------------------------------------------
def gm(i: str, title: str, content: str, **kw) -> Memory:
    meta = kw.pop("metadata", {})
    ents = kw.pop("entities", None)
    if ents is not None:
        meta = dict(meta, decomposition={"entities": ents})
    kw.setdefault("created_at", "2026-01-01T00:00:00+00:00")
    kw.setdefault("updated_at", kw["created_at"])
    return Memory(
        id=f"skrs-r05-g{i}", title=title, content=content, metadata=meta, **kw
    )


def gid(i):
    return f"skrs-r05-g{i}"


graph_memories = [
    gm(
        "01",
        "Alpha original",
        "alpha",
        entities=["Ali Al-Zaidi", "Zürich", "skgateway"],
        layer=MemoryLayer.LONG,
        tags=["alpha", "beta"],
        source="chat",
        context_tag="public",
        created_at="2026-01-10T00:00:00+00:00",
    ),
    gm(
        "02",
        "Alpha revised",
        "alpha v2",
        entities=["Ali Al-Zaidi", "C++ (v2.0)"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-02-10T00:00:00+00:00",
        parent_id=gid("01"),
        metadata={"parent_relation": "supersedes"},
    ),
    gm(
        "03",
        "Alpha final",
        "alpha v3",
        entities=["Ali Al-Zaidi", "O'Brien"],
        tags=["alpha", "gamma"],
        source="import",
        context_tag="team",
        created_at="2026-03-10T00:00:00+00:00",
        parent_id=gid("02"),
        metadata={"parent_relation": "supersedes"},
    ),
    gm(
        "04",
        "Beta promoted",
        "beta",
        entities=["skgateway", "東京 Tower"],
        layer=MemoryLayer.MID,
        tags=["beta"],
        source="seed",
        context_tag="public",
        created_at="2026-02-01T00:00:00+02:00",
        parent_id=gid("01"),
        metadata={"parent_relation": "promoted_from"},
    ),
    gm(
        "05",
        "Beta chunk",
        "beta chunk",
        entities=["skgateway routing", "Casey"],
        tags=["beta"],
        source="chat",
        context_tag="private",
        created_at="2026-02-02T00:00:00Z",
        parent_id=gid("04"),
        metadata={"parent_relation": "chunk_of"},
    ),
    gm(
        "06",
        "Gamma extends",
        "gamma",
        entities=["Casey", "invoice"],
        tags=["gamma"],
        source="chat",
        context_tag="public",
        created_at="2026-04-01T12:00:00-05:00",
        parent_id=gid("05"),
        metadata={"parent_relation": "extends"},
    ),
    gm(
        "07",
        "Delta related",
        "delta",
        entities=["Zürich", "naïve café"],
        tags=["delta", "alpha"],
        source="import",
        context_tag="public",
        created_at="2026-04-02T00:00:00+00:00",
        related_ids=[gid("01"), gid("06"), "skrs-r05-gMISSING"],
    ),
    gm(
        "08",
        "Epsilon valid window",
        "epsilon",
        entities=["Ali Al-Zaidi"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-01-01T00:00:00+00:00",
        valid_from="2026-03-01T00:00:00+00:00",
        valid_to="2026-06-01T02:00:00+02:00",
    ),
    gm(
        "09",
        "Zeta closed early",
        "zeta",
        entities=["Ali Al-Zaidi"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-01-01T00:00:00+00:00",
        valid_to="2026-02-01T00:00:00+00:00",
    ),
    gm(
        "10",
        "Eta starts late",
        "eta",
        entities=["Ali Al-Zaidi"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-01-01T00:00:00+00:00",
        valid_from="2026-09-01T00:00:00+00:00",
    ),
    gm("11", "Theta no entities", "short", tags=[], source="", context_tag=""),
    gm(
        "12",
        "Iota boundary",
        "iota",
        entities=["Boundary Entity"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-06-01T00:00:00+00:00",
        valid_from="2026-06-01T00:00:00+00:00",
    ),
    gm(
        "13",
        "Kappa offset boundary",
        "kappa",
        entities=["Boundary Entity"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-05-31T22:00:00-02:00",
    ),
    gm(
        "14",
        "Lambda closed at boundary",
        "lambda",
        entities=["Boundary Entity"],
        tags=["alpha"],
        source="chat",
        context_tag="public",
        created_at="2026-01-01T00:00:00Z",
        valid_to="2026-06-01T00:00:00+00:00",
    ),
    gm(
        "15",
        "Mu mixed offset sup child",
        "mu",
        entities=["Zürich"],
        tags=["beta"],
        source="chat",
        context_tag="public",
        created_at="2026-06-01T01:00:00+02:00",
        parent_id=gid("14"),
        metadata={"parent_relation": "supersedes"},
    ),
    gm(
        "16",
        "Nu promoted deep",
        "nu",
        entities=["skgateway"],
        tags=["beta"],
        source="chat",
        context_tag="public",
        created_at="2026-05-01T00:00:00Z",
        parent_id=gid("05"),
        metadata={"parent_relation": "promoted_from"},
    ),
    gm(
        "17",
        "Xi hostile ' $$ \" name",
        "xi",
        entities=["x') MATCH (n) DETACH DELETE n //", "$$ ) AS (a agtype); --"],
        tags=["x'] OR 1=1"],
        source="chat",
        context_tag="public",
    ),
]
for i in range(18, 30):
    graph_memories.append(
        gm(
            f"{i}",
            f"Filler {i:02d}",
            f"filler {i}",
            entities=["skgateway", f"Filler Entity {i}"],
            tags=["alpha" if i % 2 else "beta"],
            source="chat",
            context_tag="public",
            created_at=f"2026-05-{i:02d}T00:00:00+00:00",
        )
    )


def agtype(v):
    if isinstance(v, str):
        t = v.strip()
        for suffix in ("::vertex", "::edge", "::path"):
            if t.endswith(suffix):
                t = t[: -len(suffix)]
        try:
            return json.loads(t)
        except ValueError:
            return None
    return v


def build_graph() -> AGEGraphBackend:
    be = AGEGraphBackend.__new__(AGEGraphBackend)
    be.dsn = DSN
    be.agent = "skrs-r05"
    be.graph = GRAPH
    be._conn_obj = None
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("LOAD 'age'")
        c.execute("SET search_path = ag_catalog, public")
        if c.execute(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name::text = %s", (GRAPH,)
        ).fetchone():
            c.execute("SELECT ag_catalog.drop_graph(%s, true)", (GRAPH,))
    be.ensure_graph()
    for m in graph_memories:
        assert be.index_memory(m), m.id
    return be


def dump_graph(be: AGEGraphBackend) -> dict:
    nodes = []
    for (v,) in be._cypher_strict(
        "MATCH (m:Memory) RETURN m ORDER BY id(m)", cols="v agtype"
    ):
        props = agtype(v)["properties"]
        nodes.append({k: props.get(k) for k in props})
    entities = []
    for r in be._cypher_strict(
        "MATCH (m:Memory)-[e:MENTIONS]->(n:Entity) RETURN m.id, n.name ORDER BY id(e)",
        cols="a agtype, b agtype",
    ):
        entities.append([agtype(r[0]), agtype(r[1])])
    links = []
    for rel in ("RELATED_TO", "SUPERSEDES", "PROMOTED_FROM", "CHUNK_OF", "EXTENDS"):
        for r in be._cypher_strict(
            f"MATCH (a:Memory)-[e:{rel}]->(b:Memory) RETURN a.id, b.id, e.valid_from, e.valid_to, e.recorded_at ORDER BY id(e)",
            cols="a agtype, b agtype, f agtype, t agtype, r agtype",
        ):
            links.append(
                {
                    "rel": rel,
                    "a": agtype(r[0]),
                    "b": agtype(r[1]),
                    "valid_from": agtype(r[2]),
                    "valid_to": agtype(r[3]),
                    "recorded_at": agtype(r[4]),
                }
            )
    return {"nodes": nodes, "mentions": entities, "links": links}


def _ts(as_of):
    """The instant an as_of case evaluates at (None means now, as the backend)."""
    return parse_ts(as_of) if as_of else datetime.now(timezone.utc)


def parsed_reference(corpus: dict, method: str, args: dict):
    """The same rule evaluated on PARSED instants (skmemory.validity.parse_ts).

    The backend compares stored timestamp STRINGS in Cypher for these reads, which
    is wrong at the boundary when offsets are mixed (``+02:00`` against ``Z``).
    The Rust arm compares instants, so it is held to this reference, and every
    case where the backend's own answer differs is flagged ``lexical_divergence``.
    """
    t = _ts(args.get("as_of"))
    nodes, links = corpus["nodes"], corpus["links"]
    by_id = {n["id"]: n for n in nodes}
    pt = lambda v: parse_ts(v) if isinstance(v, str) else None  # noqa: E731

    def superseded():
        return {
            ln["b"]
            for ln in links
            if ln["rel"] == "SUPERSEDES"
            and ln["valid_to"] is not None
            and pt(ln["valid_to"])
            and pt(ln["valid_to"]) <= t
        }

    def born(n):
        c = pt(n.get("created_at"))
        return c is None or c <= t

    def window(ln):
        f, e = pt(ln["valid_from"]), pt(ln["valid_to"])
        return (f is None or f <= t) and (e is None or e > t)

    if method == "currently_valid_memories":
        sup = superseded()
        rows = [n for n in nodes if born(n) and n["id"] not in sup][: args["limit"]]
        return [
            {k: r.get(k) for k in ("id", "title", "layer", "created_at")} for r in rows
        ]
    if method == "is_currently_valid":
        n = by_id.get(args["id"])
        return bool(n) and args["id"] not in superseded() and born(n)
    if method == "neighbors":
        seen, out = set(), []
        for rel in ("RELATED_TO", "SUPERSEDES", "PROMOTED_FROM", "CHUNK_OF", "EXTENDS"):
            for ln in links:
                if ln["rel"] != rel or not window(ln):
                    continue
                other = (
                    ln["b"]
                    if ln["a"] == args["id"]
                    else ln["a"]
                    if ln["b"] == args["id"]
                    else None
                )
                if other and other != args["id"] and other not in seen:
                    seen.add(other)
                    n = by_id[other]
                    out.append(
                        {
                            "id": other,
                            "title": n.get("title"),
                            "layer": n.get("layer"),
                            "distance": 1,
                        }
                    )
        return sorted(out, key=lambda r: r["title"] or "")[:50]
    if method == "supersedes_chain":
        chain, visited, frontier = [], {args["id"]}, [(args["id"], 0)]
        while frontier:
            cur, dist = frontier.pop(0)
            if dist >= 10:
                continue
            for ln in links:
                if (
                    ln["rel"] == "SUPERSEDES"
                    and ln["a"] == cur
                    and window(ln)
                    and ln["b"] not in visited
                ):
                    visited.add(ln["b"])
                    n = by_id[ln["b"]]
                    chain.append(
                        {
                            "id": ln["b"],
                            "title": n.get("title"),
                            "layer": n.get("layer"),
                            "distance": dist + 1,
                        }
                    )
                    frontier.append((ln["b"], dist + 1))
        return sorted(chain, key=lambda r: r["distance"])
    raise KeyError(method)


def stub(d):
    out = {k: d.get(k) for k in ("id", "title", "layer")}
    dist = d.get("distance", d.get("depth"))
    if dist is not None:
        out["distance"] = dist
    return out


def graph_cases(be: AGEGraphBackend, corpus: dict) -> list[dict]:
    out = []

    def add(name, method, args, filters=None, ordered=False):
        entry = {
            "name": name,
            "method": method,
            "args": args,
            "filters": filters or {},
            "ordered": ordered,
        }
        f = dict(filters or {})
        kw = {}
        if "layer" in f:
            kw["layer"] = MemoryLayer(f["layer"])
        for k in ("tags", "context_tag", "source", "as_of"):
            if k in f:
                kw[k] = f[k]
        try:
            if method in ("search_by_entity", "search_by_entity_tokens"):
                rows = getattr(be, method)(args["query"], args["limit"], **kw)
                entry["expect"] = [
                    {k: r.get(k) for k in ("id", "title", "layer", "entity")}
                    for r in rows
                ]
            elif method == "currently_valid_memories":
                rows = be.currently_valid_memories(args.get("as_of"), args["limit"])
                entry["expect"] = [
                    {k: r.get(k) for k in ("id", "title", "layer", "created_at")}
                    for r in rows
                ]
            elif method == "is_currently_valid":
                entry["expect"] = be.is_currently_valid(args["id"], args.get("as_of"))
            elif method == "neighbors":
                entry["expect"] = [
                    stub(r) for r in be.neighbors(args["id"], args.get("as_of"))
                ]
            elif method == "supersedes_chain":
                entry["expect"] = [
                    stub(r) for r in be.supersedes_chain(args["id"], args.get("as_of"))
                ]
            elif method == "get_related":
                entry["expect"] = [
                    stub(r) for r in be.get_related(args["id"], args["depth"])
                ]
            elif method == "get_lineage":
                entry["expect"] = [stub(r) for r in be.get_lineage(args["id"])]
            if (
                method
                in (
                    "currently_valid_memories",
                    "is_currently_valid",
                    "neighbors",
                    "supersedes_chain",
                )
                and args.get("as_of")
                and parse_ts(args["as_of"]) is None
            ):
                # The backend compares text, so garbage "works" (it sorts after every date).
                # The Rust arm refuses it as a typed BadAsOf, like every other as_of.
                entry["rust_expect_error"] = "BadAsOf"
            elif (
                method
                in (
                    "currently_valid_memories",
                    "is_currently_valid",
                    "neighbors",
                    "supersedes_chain",
                )
                and "expect" in entry
            ):
                entry["expect_parsed"] = parsed_reference(corpus, method, args)
                entry["lexical_divergence"] = entry["expect_parsed"] != entry["expect"]
        except Exception as e:  # noqa: BLE001
            entry["expect_error"] = type(e).__name__
        out.append(entry)

    for name, q, limit, f in [
        ("ent_whole", "Ali Al-Zaidi", 50, {}),
        ("ent_substring_ci", "al-zaidi", 50, {}),
        ("ent_upper", "ZÜRICH", 50, {}),
        ("ent_unicode", "東京", 50, {}),
        ("ent_regex_chars", "C++ (v2.0)", 50, {}),
        ("ent_quote", "O'Brien", 50, {}),
        ("ent_hostile_a", "x') MATCH (n) DETACH DELETE n //", 50, {}),
        ("ent_hostile_b", "$$ ) AS (a agtype); --", 50, {}),
        ("ent_dot_not_wildcard", "v2.0", 50, {}),
        ("ent_wildcard_literal", ".*", 50, {}),
        ("ent_empty", "", 50, {}),
        ("ent_blank", "   ", 50, {}),
        ("ent_none", "zzznotanentity", 50, {}),
        ("ent_layer", "Ali", 50, {"layer": "long-term"}),
        ("ent_tags_and", "Ali", 50, {"tags": ["alpha", "gamma"]}),
        ("ent_tag_hostile", "skgateway", 50, {"tags": ["x'] OR 1=1"]}),
        ("ent_ctx_one", "Ali", 50, {"context_tag": "team"}),
        ("ent_ctx_any", "Ali", 50, {"context_tag": ["team", "private"]}),
        ("ent_ctx_empty_list", "Ali", 50, {"context_tag": []}),
        ("ent_source", "Ali", 50, {"source": "import"}),
        ("ent_source_empty", "Ali", 50, {"source": ""}),
        (
            "ent_combo",
            "Ali",
            50,
            {
                "layer": "mid-term",
                "tags": ["alpha"],
                "source": "chat",
                "context_tag": "public",
            },
        ),
        ("ent_asof_date", "Ali", 50, {"as_of": "2026-04-10"}),
        ("ent_asof_boundary", "Boundary", 50, {"as_of": "2026-06-01T00:00:00Z"}),
        ("ent_asof_offset", "Boundary", 50, {"as_of": "2026-06-01T01:00:00+01:00"}),
        ("ent_asof_before", "Ali", 50, {"as_of": "2025-01-01"}),
        ("ent_asof_future", "Ali", 50, {"as_of": "2030-01-01"}),
        ("ent_asof_limit2", "Ali", 2, {"as_of": "2026-04-10"}),
        ("ent_limit1", "skgateway", 1, {}),
    ]:
        add(name, "search_by_entity", {"query": q, "limit": limit}, f)
    for name, q, limit, f in [
        ("tok_multiword", "what did Casey say about the invoice", 20, {}),
        ("tok_whole_then_tokens", "Ali Al-Zaidi", 20, {}),
        ("tok_stopwords_only", "what about this", 20, {}),
        ("tok_short_words", "a ab abc", 20, {}),
        ("tok_unicode_mixed", "Zürich café naïve", 20, {}),
        ("tok_identifier", "skgateway routing decision", 20, {}),
        ("tok_limit1", "skgateway routing decision", 1, {}),
        ("tok_limit3", "Casey invoice skgateway", 3, {}),
        ("tok_dupe_tokens", "skgateway skgateway skgateway", 20, {}),
        ("tok_hostile", "'); MATCH (n) DETACH DELETE n; $$ skgateway", 20, {}),
        ("tok_layer", "skgateway Casey", 20, {"layer": "mid-term"}),
        ("tok_asof", "Ali Zaidi Boundary", 20, {"as_of": "2026-06-01T00:00:00Z"}),
        ("tok_empty", "", 20, {}),
    ]:
        add(name, "search_by_entity_tokens", {"query": q, "limit": limit}, f)
    # temporal reads
    for name, as_of in [
        ("cv_now", None),
        ("cv_2026_01", "2026-01-15"),
        ("cv_2026_02", "2026-02-15"),
        ("cv_2026_03", "2026-03-15"),
        ("cv_2026_04", "2026-04-15"),
        ("cv_boundary_z", "2026-06-01T00:00:00Z"),
        ("cv_boundary_offset", "2026-06-01T01:00:00+01:00"),
        ("cv_before_all", "2025-01-01"),
        ("cv_future", "2030-01-01"),
        ("cv_naive", "2026-04-10T05:00:00"),
        ("cv_mixed_offset_edge", "2026-05-31T23:30:00Z"),
    ]:
        add(name, "currently_valid_memories", {"as_of": as_of, "limit": 200})
    add("cv_limit5", "currently_valid_memories", {"as_of": "2026-12-01", "limit": 5})
    add("cv_bad", "currently_valid_memories", {"as_of": "now-ish", "limit": 5})
    for mid in ("01", "02", "03", "09", "10", "MISSING"):
        for tag, as_of in (
            ("now", None),
            ("jan", "2026-01-15"),
            ("mid", "2026-02-20"),
            ("late", "2026-12-01"),
        ):
            add(
                f"valid_{mid}_{tag}",
                "is_currently_valid",
                {"id": gid(mid), "as_of": as_of},
            )
    for mid in ("01", "03", "05", "06", "07", "14", "15", "11", "MISSING"):
        for tag, as_of in (
            ("now", None),
            ("past", "2026-01-15"),
            ("mid", "2026-04-20"),
            ("future", "2030-01-01"),
        ):
            add(
                f"nb_{mid}_{tag}",
                "neighbors",
                {"id": gid(mid), "as_of": as_of},
                ordered=True,
            )
    for mid in ("03", "02", "15", "05", "01"):
        for tag, as_of in (
            ("now", None),
            ("past", "2026-01-15"),
            ("feb", "2026-02-20"),
            ("mar", "2026-03-20"),
            ("future", "2030-01-01"),
        ):
            add(
                f"chain_{mid}_{tag}",
                "supersedes_chain",
                {"id": gid(mid), "as_of": as_of},
                ordered=True,
            )
    add(
        "chain_boundary",
        "supersedes_chain",
        {"id": gid("15"), "as_of": "2026-05-31T23:00:00Z"},
        ordered=True,
    )
    add(
        "chain_boundary_after",
        "supersedes_chain",
        {"id": gid("15"), "as_of": "2026-05-31T23:00:01Z"},
        ordered=True,
    )
    for mid in ("01", "03", "05", "06", "07", "16", "MISSING"):
        for d in (1, 2, 5, 9, 0):
            add(
                f"rel_{mid}_d{d}",
                "get_related",
                {"id": gid(mid), "depth": d},
                ordered=True,
            )
        add(f"lin_{mid}", "get_lineage", {"id": gid(mid)}, ordered=True)
    return out


def main() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM memories WHERE id LIKE 'skrs-r05-%'")
        for b in bodies:
            cols = (
                "id, agent, layer, role, title, content, summary, tags, source, created_at, "
                "updated_at, memory_json"
            )
            vals = [
                b["id"],
                b["agent"],
                b["layer"],
                b["role"],
                b["title"],
                b["content"],
                b["summary"] or "",
                b["tags"],
                b["source"],
                b["created_at"],
                b["updated_at"],
                json.dumps(b["memory_json"]),
            ]
            ph = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s::jsonb"
            if b["v2"]:
                cols += ", kind, outcome, event_at, valid_from, valid_to, supersedes_id, source_tier, schema_version"
                ph += ",%s,%s,%s::timestamptz,%s::timestamptz,%s::timestamptz,%s,%s,%s"
                vals += [
                    b["kind"],
                    b["outcome"],
                    b["event_at"],
                    b["valid_from"],
                    b["valid_to"],
                    b["supersedes_id"],
                    b["source_tier"],
                    b["schema_version"],
                ]
            conn.execute(f"INSERT INTO memories ({cols}) VALUES ({ph})", vals)
        try:
            lex = run_lexical(conn)
        finally:
            conn.execute("DELETE FROM memories WHERE id LIKE 'skrs-r05-%'")
    (OUT / "r05_lex_corpus.json").write_text(
        json.dumps(bodies, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "r05_lex_expected.json").write_text(
        json.dumps(lex, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    be = build_graph()
    try:
        corpus = dump_graph(be)
        cases = graph_cases(be, corpus)
    finally:
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("LOAD 'age'")
            c.execute("SET search_path = ag_catalog, public")
            c.execute("SELECT ag_catalog.drop_graph(%s, true)", (GRAPH,))
    (OUT / "r05_graph_corpus.json").write_text(
        json.dumps(corpus, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "r05_graph_expected.json").write_text(
        json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_err = sum(1 for e in lex if "expect_error" in e)
    n_fail = sum(1 for e in lex if e.get("bm25_failed"))
    print(
        len(bodies),
        "lex rows,",
        len(lex),
        "lex cases (",
        n_err,
        "errors,",
        n_fail,
        "bm25 fallbacks),",
        len(corpus["nodes"]),
        "graph nodes,",
        len(corpus["links"]),
        "links,",
        len(cases),
        "graph cases,",
        sum(1 for e in cases if "expect_error" in e),
        "graph errors,",
        sum(1 for e in cases if e.get("lexical_divergence")),
        "lexical divergences",
    )


main()
