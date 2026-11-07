#!/usr/bin/env python3
"""Time the Python prefilter on the store the Rust bench built (SKMEM-R-07).

Usage: PYTHONPATH=<skmemory> python bench_prefilter.py <base> <runs>
Prints one JSON object of median milliseconds per scenario, same scenarios as
`tests/intent_bench.rs`: cold = `prefilter_safe` (open store, build matcher, scan),
warm = prefilter on an open store (matcher cached).
"""
import json
import statistics
import sys
import time

from skmemory import intentions as I

base, runs = sys.argv[1], int(sys.argv[2])
SHORT = "please look at the rollout notes and also alpha7 gamma then check delta7 and epsilon things " * 3
HIT_SCOPE_FAIL = "talk about alpha9 beta today"
LONG = ("lorem ipsum dolor sit amet alpha3 gamma " * 3000) + " end"


def med(f):
    xs = []
    for _ in range(runs):
        t = time.perf_counter()
        f()
        xs.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(xs), 3)


out = {}
for name, text in [("short_nomatch", SHORT), ("phrase_hit_scope_fail", HIT_SCOPE_FAIL), ("long_100kb", LONG)]:
    out[f"cold_{name}"] = med(lambda: I.prefilter_safe(base, text, {}))
    st = I.IntentionStore(base)
    st.prefilter(text, {})
    out[f"warm_{name}"] = med(lambda: st.prefilter(text, {}, deadline_s=5.0))
    st.close()
print(json.dumps(out))
