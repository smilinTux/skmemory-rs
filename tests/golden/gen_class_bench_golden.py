#!/usr/bin/env python3
"""Record the Python class bench numbers for the vendored v1 fixtures.

Writes tests/fixtures/eval-v1/class-bench-python.json from the CURRENT Python
skmemory (`skmemory.eval.class_bench`, hashed offline embedding, no network, no
database). Read-only against the skmemory checkout.

    env -u SKMEMORY_PG_DSN SKMEMORY_EMBED_URL=http://127.0.0.1:1/api/embed \
      SKCAPSTONE_HOME=<scratch> SKMEMORY_HOME=<scratch> SKMEMORY_RETRIEVAL_LOG=off \
      PYTHONDONTWRITEBYTECODE=1 python tests/golden/gen_class_bench_golden.py

Recorded: per-class recall@5 / mrr / abstain numbers for every strategy, the
candidate's ranked page for every query, the time window the temporal arm
parses for every query, and probes for the hashed embedding (blake2b bucket).
"""

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import skmemory
from skmemory.eval import class_bench as cb
from skmemory.eval import fusion_bench as fb
from skmemory.eval.fixtures import load_corpus, load_queries
from skmemory.fusion import parse_time_window

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "eval-v1" / "class-bench-python.json"
NOW = datetime.fromisoformat(cb.FIXTURE_NOW)


def main() -> None:
    res = cb.run()
    corpus = load_corpus()
    queries = load_queries()
    pages, windows = {}, {}
    with tempfile.TemporaryDirectory(prefix="class-bench-gold-") as home, cb._fusion_env():
        store = cb.build_production_store(home, corpus)
        for q in queries:
            ctx = None if q.reader == "@chef" else "@public"
            pages[q.id] = [m.id for m in store.search(q.query, limit=10, context_tag=ctx, as_of=q.as_of)]
            w = parse_time_window(q.query, now=NOW, strict=True)
            windows[q.id] = list(w) if w else None
    by_class = {
        c: {
            "queries": e["queries"],
            "strategies": {n: s for n, s in e["strategies"].items()},
        }
        for c, e in res["by_class"].items()
    }
    probes = [
        {"concept": t, "bucket": int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big") % fb._EMBED_DIM}
        for t in ["pg", "vectorsearch", "durability", "skap08", "prb", "6f069c5e", "a", "restore", "x" * 200]
    ]
    try:
        sha = subprocess.run(
            ["git", "-C", str(Path(skmemory.__file__).parent.parent), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    out = {
        "meta": {"skmemory_commit": sha, "threshold": res["threshold"], "embed_mode": res["embed_mode"],
                 "fixture_now": cb.FIXTURE_NOW, "corpus_tokens": res["corpus_tokens"]},
        "by_class": by_class,
        "candidate_pages": pages,
        "time_windows": windows,
        "bucket_probes": probes,
        "failures": res["failures"],
        "leak_count": res["leak_count"],
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
