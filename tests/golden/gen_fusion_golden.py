#!/usr/bin/env python3
"""Generate tests/golden/fusion_parity.json from the CURRENT Python skmemory.

Every case is synthetic (invented ids, invented words, fixed seed). Nothing here
reads an agent store, a database or the network. The expected values are the
output of skmemory's own functions:

    skmemory.fusion        run_arm, reciprocal_rank_fusion, rank_interleave,
                           pinned_rrf, fuse, route_query
    skmemory.rerank        RareTermReranker via MemoryStore._reranker (env aware)
    skmemory.store         _demote_superseded and the whole MemoryStore.search path
    skmemory.tiers         apply_tier_cap
    skmemory.procedures    order_procedures
    skmemory.decay         rerank, configured_channel

The ``pipeline`` section drives a real ``MemoryStore.search`` over fake backends,
so it covers the order the store applies the stages in (fuse, rerank, demote,
decay, demote, tier cap, procedure order) rather than a replay of it.

Run (read-only against the skmemory checkout, no bytecode written there):

    PYTHONDONTWRITEBYTECODE=1 ~/.skenv/bin/python tests/golden/gen_fusion_golden.py

The skmemory commit used is recorded in the output's ``meta``.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OUT = HERE / "fusion_parity.json"

# Hermetic: no retrieval log, no agent home, and the process env is rebuilt per case.
_HOME = tempfile.mkdtemp(prefix="skmem-fusion-golden-")
os.environ["SKCAPSTONE_HOME"] = _HOME
os.environ["SKMEMORY_RETRIEVAL_LOG"] = "off"

from skmemory import decay as pydecay  # noqa: E402
from skmemory import fusion as F  # noqa: E402
from skmemory.models import EmotionalSnapshot, Memory  # noqa: E402
from skmemory.procedures import order_procedures  # noqa: E402
from skmemory.query_sanitizer import sanitize_query  # noqa: E402
from skmemory.store import MemoryStore, _demote_superseded  # noqa: E402
from skmemory.tiers import apply_tier_cap  # noqa: E402

AS_OF = "2026-11-01T12:00:00+00:00"
ENV_KEYS = [
    "SKMEMORY_FUSION_STRATEGY",
    "SKMEMORY_FUSION_PIN",
    "SKMEMORY_FUSION_ROUTE",
    "SKMEMORY_FUSION_VECTOR_PIN",
    "SKMEMORY_FUSION_FILL",
    "SKMEMORY_TIER_CAP",
    "SKMEMORY_SEARCH_RERANK",
    "SKMEMORY_SEARCH_RERANK_DIVISOR",
    "SKMEMORY_DECAY",
    "SKMEMORY_GRAPH_ARM",
    "SKMEMORY_SEARCH_FUSION",
]


class Env:
    """Set exactly these SKMEMORY_* variables for one case, restore afterwards."""

    def __init__(self, env):
        self.env = env or {}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(self.env)

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- arms


def arm_from_spec(name, spec):
    """spec: list of ids | {"raise": msg} | None (unattached) | "disabled"."""
    if spec == "disabled":
        return F.run_arm(name, lambda: [], disabled="off by config")
    if spec is None:
        return F.run_arm(name, None)
    if isinstance(spec, dict):

        def boom():
            raise RuntimeError(spec["raise"])

        return F.run_arm(name, boom)
    return F.run_arm(name, lambda: list(spec))


def outcome_json(a):
    return {"name": a.name, "status": a.status, "ids": list(a.ids)}


# --------------------------------------------------------------------------- routes

ROUTE_QUERIES = [
    ("prb-6f069c5e remediation", None),
    ("zioap01", None),
    ("restart skgateway_pool", None),
    ("grafana.example.net certificate", None),
    ("what happened in 2026-10-16", None),
    ("meeting on the 5th", None),
    ("sync notes 2026", None),
    ("how do we restore the replica", None),
    ("did the rollout finish", None),
    ("rollout finished?", None),
    ("the plan for the cluster and the rebuild of it", None),
    ("cluster rebuild plan", None),
    ("postgres rebuild", None),
    ("", None),
    ("---", None),
    ("a b", None),
    ("tell me about gateway pools", None),
    ("Remind me about the rotation", None),
    ("gateway pool ceiling", {"gateway": 2, "pool": 0, "ceiling": 40}),
    ("gateway pool ceiling", {"gateway": 0, "pool": 9, "ceiling": 40}),
    ("gateway pool ceiling", {"ceiling": 3}),
    ("what is the ceiling", {"ceiling": 2}),
    ("what is the ceiling", {"ceiling": 4}),
    ("one two three four five six seven eight nine ten", {"nine": 1}),
    ("one two three four five six seven eight nine ten", {"eight": 1}),
    ("the of to in on at for with", None),
    ("v1.2.3 release", None),
    ("e-mail rules", None),
    ("a-b-c", None),
    ("node-07 outage", None),
    ("first_second", None),
    ("why is it so", None),
    ("is", None),
    ("QUICK Brown Fox", None),
    ("what 2nd", None),
]
ROUTE_ENVS = [
    {},
    {"SKMEMORY_FUSION_ROUTE": "lexical"},
    {"SKMEMORY_FUSION_ROUTE": "vector"},
    {"SKMEMORY_FUSION_ROUTE": "auto"},
    {"SKMEMORY_FUSION_ROUTE": "bogus"},
]


def route_cases():
    out = []
    for q, df in ROUTE_QUERIES:
        for env in ROUTE_ENVS if q in ("zioap01", "how do we restore the replica") else [{}]:
            with Env(env):
                d = F.route_query(q, (lambda t, df=df: df.get(t, 0)) if df is not None else None)
            out.append({"query": q, "df": df, "env": env, "cls": d.cls, "reason": d.reason})
    return out


# --------------------------------------------------------------------------- fuse

POOL = [f"m-{i:02d}" for i in range(24)]


def rand_arm_spec(rng):
    r = rng.random()
    if r < 0.07:
        return None
    if r < 0.12:
        return "disabled"
    if r < 0.19:
        return {"raise": "dead backend"}
    if r < 0.27:
        return []
    n = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, 12, 30])
    base = rng.sample(POOL, min(n, len(POOL)))
    if n == 30:
        base = base + [f"x-{i:02d}" for i in range(30 - len(base))]
    if rng.random() < 0.25:  # a duplicate inside one arm
        base.insert(rng.randrange(len(base) + 1), rng.choice(base))
    return base


FUSE_ENVS = [
    {},
    {},
    {"SKMEMORY_FUSION_VECTOR_PIN": "0"},
    {"SKMEMORY_FUSION_VECTOR_PIN": "2"},
    {"SKMEMORY_FUSION_VECTOR_PIN": "5"},
    {"SKMEMORY_FUSION_VECTOR_PIN": "x"},
    {"SKMEMORY_FUSION_PIN": "0"},
    {"SKMEMORY_FUSION_PIN": "1"},
    {"SKMEMORY_FUSION_PIN": "5"},
    {"SKMEMORY_FUSION_PIN": "-2"},
    {"SKMEMORY_FUSION_PIN": "y"},
    {"SKMEMORY_FUSION_FILL": "rrf"},
    {"SKMEMORY_FUSION_FILL": " RRF "},
    {"SKMEMORY_FUSION_FILL": "interleave"},
    {"SKMEMORY_FUSION_PIN": "6", "SKMEMORY_FUSION_VECTOR_PIN": "6", "SKMEMORY_FUSION_FILL": "rrf"},
]
ARM_NAMES = ["semantic", "lexical", "lexical_local", "graph", "temporal"]


def fuse_case(arm_specs, *, limit, strategy, pin_arms, route, env, k=None, name=""):
    outcomes = [arm_from_spec(n, s) for n, s in arm_specs]
    rd = F.RouteDecision(route, "golden") if route else None
    kw = {} if k is None else {"k": k}
    with Env(env):
        res = F.fuse(
            outcomes,
            limit=limit,
            strategy=strategy,
            pin_arms=tuple(pin_arms),
            route=rd,
            **kw,
        )
    return {
        "name": name,
        "arms": [{"name": n, "spec": s} for n, s in arm_specs],
        "arm_outcomes": [outcome_json(a) for a in outcomes],
        "limit": limit,
        "strategy": strategy,
        "pin_arms": list(pin_arms),
        "route": route,
        "k": k,
        "env": env,
        "ids": list(res.ids),
        "provenance": {i: dict(p) for i, p in res.provenance.items()},
        "result_route": res.route,
    }


def fuse_cases():
    out = []
    rng = random.Random(20261107)
    lex_first = ["lexical", "lexical_local"]
    local_first = ["lexical_local", "lexical"]
    # hand written
    base = [
        ("semantic", ["a", "b", "c", "d", "e", "f"]),
        ("lexical", ["x", "b", "y", "z", "a", "q"]),
        ("lexical_local", ["y", "x", "b", "w"]),
        ("graph", None),
        ("temporal", []),
    ]
    for strategy in ("pinned_rrf", "interleave", "rrf"):
        for route in (None, "identifier", "natural"):
            for limit in (1, 3, 10):
                for pins in (lex_first, local_first):
                    out.append(
                        fuse_case(
                            base,
                            limit=limit,
                            strategy=strategy,
                            pin_arms=pins,
                            route=route,
                            env={},
                            name=f"base-{strategy}-{route}-{limit}-{pins[0]}",
                        )
                    )
    # short vector arm (fewer hits than the pin) and a duplicate-only arm
    short = [("semantic", ["a", "b"]), ("lexical", ["c", "d", "e"]), ("lexical_local", ["e"])]
    for limit in (1, 3, 10):
        out.append(
            fuse_case(
                short,
                limit=limit,
                strategy="pinned_rrf",
                pin_arms=lex_first,
                route="natural",
                env={},
                name=f"short-vector-{limit}",
            )
        )
    dup = [("semantic", ["a", "a", "a", "b", "b"]), ("lexical_local", ["a", "c", "c"])]
    for route in ("natural", "identifier", None):
        out.append(
            fuse_case(
                dup,
                limit=10,
                strategy="pinned_rrf",
                pin_arms=lex_first,
                route=route,
                env={},
                name=f"duplicates-{route}",
            )
        )
    # dead arms: every arm dead, only vector dead, only lexical dead
    dead = [
        ("semantic", {"raise": "no embed server"}),
        ("lexical", {"raise": "bm25 down"}),
        ("lexical_local", {"raise": "fts down"}),
        ("graph", "disabled"),
        ("temporal", None),
    ]
    out.append(
        fuse_case(
            dead,
            limit=10,
            strategy="pinned_rrf",
            pin_arms=lex_first,
            route="natural",
            env={},
            name="all-dead",
        )
    )
    vdead = [("semantic", {"raise": "x"}), ("lexical", ["a", "b", "c", "d"]), ("lexical_local", ["b", "a"])]
    for route in ("natural", "identifier"):
        out.append(
            fuse_case(
                vdead,
                limit=10,
                strategy="pinned_rrf",
                pin_arms=lex_first,
                route=route,
                env={},
                name=f"vector-dead-{route}",
            )
        )
    nolex = [("semantic", ["a", "b", "c", "d", "e"]), ("graph", ["z", "a"])]
    for route in ("identifier", "natural", None):
        out.append(
            fuse_case(
                nolex,
                limit=10,
                strategy="pinned_rrf",
                pin_arms=lex_first,
                route=route,
                env={},
                name=f"no-lexical-{route}",
            )
        )
    # custom k
    out.append(
        fuse_case(
            base, limit=10, strategy="rrf", pin_arms=lex_first, route=None, env={}, k=10, name="k10"
        )
    )
    # random sweep
    for n in range(110):
        specs = [(nm, rand_arm_spec(rng)) for nm in ARM_NAMES]
        out.append(
            fuse_case(
                specs,
                limit=rng.choice([1, 3, 10, 30]),
                strategy=rng.choice(["pinned_rrf", "pinned_rrf", "pinned_rrf", "interleave", "rrf"]),
                pin_arms=rng.choice([lex_first, local_first]),
                route=rng.choice([None, "identifier", "natural", "natural"]),
                env=rng.choice(FUSE_ENVS),
                name=f"random-{n:03d}",
            )
        )
    return out


def score_cases():
    out = []
    rng = random.Random(99)
    for n in range(24):
        arms = {}
        for nm in rng.sample(ARM_NAMES, rng.randint(1, 4)):
            ids = rng.sample(POOL, rng.randint(1, 9))
            arms[nm] = ids
        env = rng.choice([{}, {"SKMEMORY_FUSION_FILL": "rrf"}])
        k = rng.choice([60, 60, 10, 1])
        pin = rng.choice([None, 0, 1, 3, 6])
        with Env(env):
            rrf = F.reciprocal_rank_fusion(arms, k)
            ril = F.rank_interleave(arms, k)
            pr = F.pinned_rrf(
                arms,
                k,
                pin=pin,
                pin_arms=("lexical_local", "lexical", "semantic"),
                slot_arms=("graph", "lexical"),
            )
        out.append(
            {
                "arms": arms,
                "k": k,
                "pin": pin,
                "env": env,
                "pin_arms": ["lexical_local", "lexical", "semantic"],
                "slot_arms": ["graph", "lexical"],
                "rrf": [[d, s, p] for d, s, p in rrf],
                "interleave": [[d, s, p] for d, s, p in ril],
                "pinned": [[d, s, p] for d, s, p in pr],
            }
        )
    return out


# --------------------------------------------------------------------------- rerank

WORDS = (
    "river stone gateway pool rebuild plan cluster node rotate secret lamp harbor copper "
    "ledger orbit meadow quartz signal window archive bridge pantry"
).split()
IDENTS = ["prb-6f069c5e", "zioap01", "node07", "err4041", "abcdefghijklmnop", "v2", "a1"]


def rand_text(rng, idents):
    toks = rng.sample(WORDS, rng.randint(3, 8))
    for i in idents:
        toks.insert(rng.randrange(len(toks) + 1), i)
    return " ".join(toks)


def rerank_cases():
    out = []
    rng = random.Random(7)

    def add(query, docs, env, name):
        with Env(env):
            r = MemoryStore._reranker(None)
            o = r.rerank(query, docs)
        out.append(
            {
                "name": name,
                "query": query,
                "docs": docs,
                "env": env,
                "order": list(o.order),
                "status": o.status,
                "boost_terms": list(o.boost_terms),
                "boosted": list(o.boosted),
            }
        )

    docs = [
        "gateway pool rebuild plan",
        "node cluster plan for the rebuild",
        "terse prb-6f069c5e record",
        "other note about the pool",
        "another zioap01 prb-6f069c5e line",
        "meadow quartz",
    ]
    add("per node rebuild plan prb-6f069c5e", docs, {}, "rare-identifier")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK": "off"}, "off")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK": "FALSE"}, "off-false")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "2.0"}, "div2")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "25"}, "div25")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "0.5"}, "div-half")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "0"}, "div0")
    add("per node rebuild plan prb-6f069c5e", docs, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "abc"}, "divbad")
    add("rebuild plan paraphrase only", docs, {}, "abstain-no-identifier")
    add("prb-6f069c5e", ["only one"], {}, "abstain-pool-too-small")
    add("prb-6f069c5e", [], {}, "abstain-empty")
    add("zioap01 prb-6f069c5e", docs, {}, "two-terms")
    add("abcdefghijklmnop thing", ["abcdefghijklmnop here", "x y", "z q"] + ["pad"] * 8, {}, "opaque-long")
    spread = [f"common zioap01 {i}" for i in range(6)]
    add("zioap01", spread, {}, "spread-abstains")
    for n in range(40):
        k = rng.randint(2, 20)
        qid = rng.sample(IDENTS, rng.randint(0, 2))
        query = " ".join(rng.sample(WORDS, 2) + qid)
        ds = []
        for _ in range(k):
            ids = [i for i in qid if rng.random() < 0.15]
            ds.append(rand_text(rng, ids))
        add(query, ds, rng.choice([{}, {}, {"SKMEMORY_SEARCH_RERANK_DIVISOR": "3"}]), f"random-{n:02d}")
    return out


# --------------------------------------------------------------------------- docs


def mk_memory(d):
    """Real skmemory Memory from a golden doc dict (flat keys)."""
    meta = {}
    if d.get("superseded_by_id"):
        meta["superseded_by"] = {"id": d["superseded_by_id"]}
    kw = dict(
        id=d["id"],
        title=d.get("title") or d["id"],
        content=d.get("content", ""),
        created_at=d.get("created_at", "2026-10-01T00:00:00+00:00"),
        layer=d.get("layer", "short-term"),
        source=d.get("source", "manual"),
        source_tier=d.get("source_tier", "user-stated"),
        schema_version=d.get("schema_version", 2),
        emotional=EmotionalSnapshot(intensity=d.get("intensity", 0.0)),
        kind=d.get("kind", "fact"),
        metadata=meta,
    )
    if d.get("outcome") is not None:
        kw["outcome"] = d["outcome"]
    if d.get("valid_to") is not None:
        kw["valid_to"] = d["valid_to"]
    if d.get("supersedes_id"):
        kw["supersedes_id"] = d["supersedes_id"]
    return Memory(**kw)


# --------------------------------------------------------------------------- demote


def demote_cases():
    out = []

    def add(ids, docs, name):
        found = {
            d["id"]: SimpleNamespace(
                supersedes_id=d.get("supersedes_id"),
                valid_to=d.get("valid_to"),
                metadata={"superseded_by": {"id": d["superseded_by_id"]}}
                if d.get("superseded_by_id")
                else {},
            )
            for d in docs
        }
        out.append({"name": name, "ids": ids, "docs": docs, "out": _demote_superseded(list(ids), found)})

    CL = "2026-10-20T00:00:00+00:00"
    add(["old", "new", "x"], [{"id": "old", "valid_to": CL}, {"id": "new", "supersedes_id": "old"}, {"id": "x"}], "closed-above-head")
    add(["new", "old", "x"], [{"id": "old", "valid_to": CL}, {"id": "new", "supersedes_id": "old"}, {"id": "x"}], "already-below")
    add(["x", "old", "y", "new"], [{"id": "old", "valid_to": CL}, {"id": "new", "supersedes_id": "old"}, {"id": "x"}, {"id": "y"}], "moves-down-to-head")
    add(["v1", "x", "v2", "v3"], [{"id": "v1", "valid_to": CL}, {"id": "v2", "valid_to": CL, "supersedes_id": "v1"}, {"id": "v3", "supersedes_id": "v2"}, {"id": "x"}], "chain-of-three")
    add(["old", "x"], [{"id": "old", "valid_to": CL}, {"id": "x"}, {"id": "new", "supersedes_id": "old"}], "head-not-in-pool")
    add(["old", "x", "new"], [{"id": "old", "valid_to": CL, "superseded_by_id": "new"}, {"id": "x"}, {"id": "new"}], "metadata-superseded-by")
    add(["old", "x", "new"], [{"id": "old", "superseded_by_id": "new"}, {"id": "x"}, {"id": "new"}], "open-record-not-demoted")
    add(["a", "b", "c"], [{"id": "a", "valid_to": CL, "supersedes_id": "b"}, {"id": "b", "valid_to": CL, "supersedes_id": "a"}, {"id": "c"}], "cycle")
    add(["a", "b", "c", "d"], [{"id": "a", "valid_to": CL}, {"id": "b", "valid_to": CL}, {"id": "c", "supersedes_id": "a"}, {"id": "d", "supersedes_id": "b"}], "two-pairs")
    add(["a", "z", "b", "c"], [{"id": "a", "valid_to": CL}, {"id": "b", "valid_to": CL}, {"id": "c", "supersedes_id": "a"}, {"id": "z", "supersedes_id": "b"}], "two-pairs-interleaved")
    add(["unknown", "old", "new"], [{"id": "old", "valid_to": CL}, {"id": "new", "supersedes_id": "old"}], "unknown-id-in-pool")
    add([], [], "empty")
    rng = random.Random(5)
    for n in range(40):
        ids = [f"d{i}" for i in range(rng.randint(2, 9))]
        docs = []
        for i in ids:
            d = {"id": i}
            r = rng.random()
            if r < 0.4:
                d["valid_to"] = CL
            if rng.random() < 0.4:
                d["supersedes_id"] = rng.choice(ids)
                if d["supersedes_id"] == i:
                    del d["supersedes_id"]
            if rng.random() < 0.15:
                d["superseded_by_id"] = rng.choice(ids)
            docs.append(d)
        rng.shuffle(ids)
        add(ids, docs, f"random-{n:02d}")
    return out


# --------------------------------------------------------------------------- tier cap


def tiercap_cases():
    out = []

    def add(ids, docs, limit, env, name):
        found = {d["id"]: mk_memory(d) for d in docs}
        with Env(env):
            res = apply_tier_cap(list(ids), found, limit)
        out.append({"name": name, "ids": ids, "docs": docs, "limit": limit, "env": env, "out": res})

    def doc(i, tier="user-stated", sv=2, src="manual"):
        return {"id": i, "source_tier": tier, "schema_version": sv, "source": src}

    ids = [f"t{i}" for i in range(12)]
    mixed = [doc(i, ["self-generated", "user-stated", "web", "tool-output"][n % 4]) for n, i in enumerate(ids)]
    for limit in (1, 3, 5, 10, 12, 20, 0):
        add(ids, mixed, limit, {}, f"mixed-limit-{limit}")
    capped_only = [doc(i, "web") for i in ids]
    for limit in (1, 3, 10):
        add(ids, capped_only, limit, {}, f"all-capped-{limit}")
    add(ids, [doc(i) for i in ids], 10, {}, "none-capped")
    legacy = [
        doc("l0", "self-generated", 1, "manual"),
        doc("l1", "self-generated", 1, "skwhisper"),
        doc("l2", "self-generated", 1, " Observation "),
        doc("l3", "web", 1, "web"),
        doc("l4", "user-stated", 2),
        doc("l5", "self-generated", 2),
        doc("l6", "self-generated", 0, "skwhisper"),
        doc("l7", "self-generated", 2),
    ]
    for limit in (3, 5, 8):
        add([d["id"] for d in legacy], legacy, limit, {}, f"legacy-{limit}")
    for v in ("off", "0", "FALSE", " no ", "on", "yes", "garbage"):
        add(ids, mixed, 5, {"SKMEMORY_TIER_CAP": v}, f"env-{v.strip()}")
    add(ids + ["ghost1", "ghost2"], mixed, 10, {}, "unknown-ids-uncapped")
    rng = random.Random(11)
    for n in range(40):
        k = rng.randint(1, 20)
        ids = [f"r{i}" for i in range(k)]
        docs = [
            doc(
                i,
                rng.choice(["user-stated", "tool-output", "web", "self-generated"]),
                rng.choice([1, 2, 2, 2]),
                rng.choice(["manual", "skwhisper", "web-fetch", "mcp"]),
            )
            for i in ids
        ]
        add(ids, docs, rng.choice([1, 3, 10, 5, 25]), rng.choice([{}, {}, {}, {"SKMEMORY_TIER_CAP": "off"}]), f"random-{n:02d}")
    return out


# --------------------------------------------------------------------------- procedures


def procedure_cases():
    out = []
    queries = [
        "how do I restore the replica",
        "how to rotate the key",
        "steps to rebuild",
        "steps for the failover",
        "runbook for the gateway",
        "procedure",
        "instructions to deploy",
        "instruction for deploy",
        "the way to recover",
        "how to fix it",
        "fixture setup",
        "how to avoid a mistake",
        "how did this fail",
        "why did the restore break",
        "should I restore",
        "how do I recover, do not skip steps",
        "shouldn't we restore",
        "how can we restore it, it didn't work",
        "HOW TO RESTORE",
        "restore went wrong",
        "just a plain query",
        "risky restore how to",
        "how to prepare a dangerous thing",
        "recovering how can",
        "how\tdo\nthe thing",
        "how   would we",
        "how to don't",
        "warnings about restore",
        "warning: how to restore",
    ]
    rng = random.Random(3)
    for q in queries:
        items = []
        for n in range(rng.randint(3, 8)):
            kind = rng.choice(["procedure", "procedure", "fact", "preference"])
            outcome = rng.choice(["success", "failure", None]) if kind == "procedure" else None
            items.append({"id": f"p{n}", "kind": kind, "outcome": outcome})
        mems = [mk_memory({"id": i["id"], "kind": i["kind"], "outcome": i["outcome"]}) for i in items]
        res = [m.id for m in order_procedures(mems, q)]
        out.append({"query": q, "items": items, "out": res})
    return out


# --------------------------------------------------------------------------- decay


def decay_cases():
    out = []
    rng = random.Random(13)
    for n in range(30):
        k = rng.randint(2, 9)
        docs, usage = [], {}
        for i in range(k):
            m = {
                "id": f"k{i}",
                "created_at": f"2026-{rng.randint(5, 10):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00+00:00",
                "layer": rng.choice(["short-term", "mid-term", "long-term"]),
                "intensity": rng.choice([0.0, 2.0, 5.0, 8.0, 10.0]),
            }
            docs.append(m)
            if rng.random() < 0.8:
                usage[m["id"]] = {
                    "access_count": rng.choice([0, 1, 3, 20, 200]),
                    "last_accessed": rng.choice(
                        [None, "2026-10-30T10:00:00+00:00", "2026-07-01T00:00:00Z", "2026-10-31T23:59:59.500000"]
                    ),
                }
        channel = rng.choice(["none", "recency", "reinforcement", "affect_weighted"])
        mems = [mk_memory(d) for d in docs]
        from datetime import datetime

        res = pydecay.rerank(mems, usage, channel, as_of=datetime.fromisoformat(AS_OF))
        out.append(
            {
                "docs": docs,
                "usage": usage,
                "channel": channel,
                "as_of": AS_OF,
                "out": [m.id for m in res],
            }
        )
    envs = [None, "", "1", "on", "TRUE", "yes", "0", "off", "false", "no", "disabled", "recency", "reinforcement", "affect_weighted", "none", "bogus", " Recency "]
    cfg = []
    for e in envs:
        with Env({} if e is None else {"SKMEMORY_DECAY": e}):
            cfg.append({"env": e, "channel": pydecay.configured_channel()})
    return out, cfg


# --------------------------------------------------------------------------- pipeline


class FakeVector:
    def __init__(self, mems, spec, lexical):
        self.mems = mems
        self.spec = spec
        if lexical is not None:
            self._lex = lexical
            self.search_lexical = self._search_lexical

    def search_semantic(self, query, limit=10, **kw):
        if isinstance(self.spec, dict):
            raise RuntimeError(self.spec["raise"])
        return [self.mems[i] for i in self.spec]

    def _search_lexical(self, query, limit=10, **kw):
        if isinstance(self._lex, dict):
            raise RuntimeError(self._lex["raise"])
        return [self.mems[i] for i in self._lex]

    def search_text(self, query, limit=10, **kw):
        return []


class FakePrimary:
    def __init__(self, mems, text, local, df, usage):
        self.mems = mems
        self.base_path = Path(_HOME) / "primary"
        self._text = text
        self._df = df
        self._usage = usage
        if local != "absent":
            self._local = local
            self.search_lexical_local = self._search_local
        if usage is not None:
            self.get_usage = lambda ids: {i: self._usage[i] for i in ids if i in self._usage}
        self.term_doc_freq = lambda t: self._df.get(t, 0)

    def _search_local(self, query, limit=10, **kw):
        if isinstance(self._local, dict):
            raise RuntimeError(self._local["raise"])
        return [self.mems[i] for i in self._local]

    def search_text(self, query, limit=10, **kw):
        if isinstance(self._text, dict):
            raise RuntimeError(self._text["raise"])
        return [self.mems[i] for i in self._text]

    def list_memories(self, **kw):
        return []

    def load(self, mid):
        return self.mems.get(mid)


class FakeGraph:
    def __init__(self, spec):
        self.spec = spec

    def search_by_entity_tokens(self, query, limit=10, **kw):
        if isinstance(self.spec, dict):
            raise RuntimeError(self.spec["raise"])
        return [{"id": i} for i in self.spec]


def pipeline_case(case, stats):
    docs = case["docs"]
    mems = {d["id"]: mk_memory(d) for d in docs}
    arms = case["arms"]
    sem = arms["semantic"]
    lex = arms["lexical"]
    use_pg = lex["src"] == "pg"
    vector = None
    if sem is not None or use_pg:
        vector = FakeVector(mems, sem if sem is not None else [], lex["ids"] if use_pg else None)
    primary = FakePrimary(
        mems,
        lex["ids"] if not use_pg else [],
        arms["lexical_local"] if arms["lexical_local"] is not None else "absent",
        case["df"],
        case["usage"],
    )
    graph_spec = arms["graph"]
    graph = None if graph_spec is None else FakeGraph([] if graph_spec == "disabled" else graph_spec)
    env = dict(case["env"])
    env["SKMEMORY_SEARCH_FUSION"] = "on"
    env["SKMEMORY_GRAPH_ARM"] = "off" if graph_spec == "disabled" else "on"
    q = case["query"]
    assert sanitize_query(q) == q, q
    with Env(env):
        store = MemoryStore(primary=primary, vector=vector, graph=graph)
        store._retrieval_log_dir = lambda: None
        orig = pydecay.rerank

        def fixed(results, usage, channel, **kw):
            from datetime import datetime

            return orig(results, usage, channel, as_of=datetime.fromisoformat(AS_OF), **kw)

        pydecay.rerank = fixed
        try:
            fused, report = store.search_fused(q, limit=max(case["limit"], case["limit"] * 3), return_report=True)
            if fused is None:
                # Nothing fused: store.search takes the legacy fallback path, which is
                # outside this port (see SOP). Not a golden case.
                return None
            page = store.search(q, limit=case["limit"], record_access=False)
            # which stage of the fused path decided the rerank outcome
            pool_ids = [i for i in (report.ids if report else [])]
        finally:
            pydecay.rerank = orig
    page_ids = [m.id for m in page]
    arms_out = [
        {"name": a.name, "status": a.status, "ids": list(a.ids)} for a in (report.arms if report else [])
    ]
    stats["route"][report.route] = stats["route"].get(report.route, 0) + 1
    return {
        "name": case["name"],
        "query": q,
        "limit": case["limit"],
        "env": case["env"],
        "docs": docs,
        "arms": arms,
        "df": case["df"],
        "usage": case["usage"],
        "decay_as_of": AS_OF,
        "expected": {
            "route": report.route if report else None,
            "arm_outcomes": arms_out,
            "fused_stage": pool_ids,
            "fused_returned": fused is not None,
            "page": page_ids,
        },
    }


def rand_doc(rng, i, ids_pool, idents):
    d = {
        "id": f"m-{i:02d}",
        "title": " ".join(rng.sample(WORDS, 2)),
        "content": rand_text(rng, [x for x in idents if rng.random() < 0.18]),
        "created_at": f"2026-{rng.randint(6, 10):02d}-{rng.randint(1, 28):02d}T08:00:00+00:00",
        "layer": rng.choice(["short-term", "mid-term", "long-term"]),
        "intensity": rng.choice([0.0, 0.0, 4.0, 9.0]),
        "source": rng.choice(["manual", "manual", "web-fetch", "skwhisper", "mcp"]),
        "source_tier": rng.choice(["user-stated", "user-stated", "tool-output", "web", "self-generated"]),
        "schema_version": rng.choice([2, 2, 2, 1]),
        "kind": rng.choice(["fact", "fact", "procedure", "preference"]),
    }
    if d["kind"] == "procedure":
        d["outcome"] = rng.choice(["success", "failure"])
    return d


PIPE_QUERIES = [
    "per node rebuild plan prb-6f069c5e",
    "zioap01 outage",
    "node07 restart",
    "what did we decide about the gateway pool",
    "how do we restore the cluster",
    "how to rotate the secret",
    "gateway pool",
    "rebuild plan",
    "the rebuild plan for the cluster and the node",
    "err4041",
    "why did the bridge fail",
    "steps to rebuild the archive",
]


def pipeline_cases():
    rng = random.Random(424242)
    stats = {"route": {}}
    out = []
    ENVS = [
        {},
        {},
        {},
        {"SKMEMORY_FUSION_VECTOR_PIN": "5"},
        {"SKMEMORY_FUSION_VECTOR_PIN": "0"},
        {"SKMEMORY_FUSION_PIN": "1"},
        {"SKMEMORY_FUSION_FILL": "rrf"},
        {"SKMEMORY_FUSION_STRATEGY": "interleave"},
        {"SKMEMORY_FUSION_STRATEGY": "rrf"},
        {"SKMEMORY_FUSION_ROUTE": "vector"},
        {"SKMEMORY_FUSION_ROUTE": "lexical"},
        {"SKMEMORY_SEARCH_RERANK": "off"},
        {"SKMEMORY_SEARCH_RERANK_DIVISOR": "2"},
        {"SKMEMORY_TIER_CAP": "off"},
        {"SKMEMORY_DECAY": "affect_weighted"},
        {"SKMEMORY_DECAY": "recency"},
        {"SKMEMORY_DECAY": "reinforcement", "SKMEMORY_FUSION_STRATEGY": "interleave"},
    ]
    for n in range(90):
        nd = rng.randint(6, 18)
        idents = rng.sample(IDENTS[:5], 2)
        docs = [rand_doc(rng, i, None, idents) for i in range(nd)]
        # supersession pairs
        for _ in range(rng.choice([0, 1, 2, 3])):
            a, b = rng.sample(range(nd), 2)
            docs[a]["valid_to"] = "2026-10-15T00:00:00+00:00"
            docs[b]["supersedes_id"] = docs[a]["id"]
        if rng.random() < 0.2:
            a, b = rng.sample(range(nd), 2)
            docs[a]["valid_to"] = "2026-10-15T00:00:00+00:00"
            docs[a]["superseded_by_id"] = docs[b]["id"]
        ids = [d["id"] for d in docs]

        def ranked():
            k = rng.randint(0, nd)
            r = rng.sample(ids, k)
            if r and rng.random() < 0.2:
                r.insert(rng.randrange(len(r) + 1), rng.choice(r))
            return r

        def spec(p_none=0.1, p_raise=0.1):
            r = rng.random()
            if r < p_none:
                return None
            if r < p_none + p_raise:
                return {"raise": "dead"}
            return ranked()

        q = rng.choice(PIPE_QUERIES)
        # plant the identifiers the query carries into some docs
        for tok in ("prb-6f069c5e", "zioap01", "node07", "err4041"):
            if tok in q:
                for d in rng.sample(docs, rng.randint(1, 2)):
                    d["content"] += " " + tok
        sem = spec(0.15, 0.1)
        lexsrc = "pg" if (sem is not None and rng.random() < 0.5) else "primary"
        if sem is None:
            lexsrc = "primary"
        lexl = rng.choice([None, "x", "x", "x"])
        arms = {
            "semantic": sem,
            "lexical": {"src": lexsrc, "ids": spec(0.0, 0.1)},
            "lexical_local": None if lexl is None else spec(0.0, 0.1),
            "graph": rng.choice([None, "disabled", ranked(), ranked()]),
            "temporal": [],
        }
        df = {t: rng.choice([0, 1, 2, 3, 5]) for t in q.split() if rng.random() < 0.3}
        usage = None
        env = rng.choice(ENVS)
        if rng.random() < 0.6:
            usage = {
                i: {
                    "access_count": rng.choice([0, 1, 4, 50]),
                    "last_accessed": rng.choice([None, "2026-10-30T10:00:00+00:00"]),
                }
                for i in ids
                if rng.random() < 0.7
            }
        case = {
            "name": f"pipeline-{n:03d}",
            "docs": docs,
            "arms": arms,
            "df": df,
            "usage": usage,
            "env": env,
            "query": q,
            "limit": rng.choice([1, 3, 10, 10]),
        }
        got = pipeline_case(case, stats)
        if got is not None:
            out.append(got)
    return out, stats


# --------------------------------------------------------------------------- main


def main():
    sk = Path(F.__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(sk), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    pipe, stats = pipeline_cases()
    dec, dec_cfg = decay_cases()
    data = {
        "meta": {
            "generator": "tests/golden/gen_fusion_golden.py",
            "skmemory_commit": commit,
            "rrf_k": F.RRF_K,
            "default_arm_cap": F.DEFAULT_ARM_CAP,
        },
        "route": route_cases(),
        "fuse": fuse_cases(),
        "scores": score_cases(),
        "rerank": rerank_cases(),
        "demote": demote_cases(),
        "tiercap": tiercap_cases(),
        "procedures": procedure_cases(),
        "decay": dec,
        "decay_config": dec_cfg,
        "pipeline": pipe,
    }
    # Coverage the Rust test relies on: the set must actually exercise each stage.
    r = data["rerank"]
    assert sum(1 for c in r if c["status"] == "applied") >= 8
    assert sum(1 for c in r if c["status"] == "abstained") >= 8
    assert sum(1 for c in data["demote"] if c["out"] != c["ids"]) >= 8
    assert sum(1 for c in data["tiercap"] if len(c["out"]) and c["out"] != c["ids"][: len(c["out"])]) >= 8
    assert sum(1 for c in data["procedures"] if c["out"] != [i["id"] for i in c["items"]]) >= 3
    assert {c["cls"] for c in data["route"]} == {"identifier", "natural"}
    assert sum(1 for c in pipe if c["expected"]["page"] != c["expected"]["fused_stage"][: len(c["expected"]["page"])]) >= 10
    assert stats["route"].get("natural", 0) >= 10 and stats["route"].get("identifier", 0) >= 10, stats
    text = json.dumps(data, indent=1, sort_keys=False, ensure_ascii=True)
    OUT.write_text(text + "\n")
    n = sum(len(data[k]) for k in ("route", "fuse", "scores", "rerank", "demote", "tiercap", "procedures", "decay", "decay_config", "pipeline"))
    print(f"wrote {OUT} ({n} cases) route_stats={stats['route']}")


if __name__ == "__main__":
    sys.exit(main())
