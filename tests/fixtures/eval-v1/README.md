# Conformance fixtures (copied, not edited)

Copied byte for byte from skmemory `skmemory/eval/fixtures/` at skmemory
origin/main `992c16fbbb08` (the Python repo is not reachable from CI):
`intentions-v1.jsonl`, `intention-transitions-v1.jsonl`, `prompts-v1.jsonl`,
`outcomes-v1.jsonl`, `outcomes-expected-v1.json`. `corpus-valid-to-v1.jsonl` is
the `(id, valid_to)` projection of `corpus-v1.jsonl`, which is all the
supersession sweep reads from an origin memory. A wrong fixture is fixed in
`gen_fixtures.py` upstream and re-copied, never edited here.

## Class bench (SKMEM-R-06B)

`corpus-v1.jsonl` (59 memories) and `queries-v1.jsonl` (53 queries) are copied
byte for byte from skmemory `skmemory/eval/fixtures/`. They are synthetic (host
names skap*/skwk*, agents marvin/astra, PR numbers 41xx; checked for IPs, e-mail
addresses, URLs, home paths and credentials: none). `class-bench-python.json` is
the output of `tests/golden/gen_class_bench_golden.py` against skmemory
`0bd94fe`: per-class numbers, the candidate page for every query, the temporal
window for every query and hashed-embedding probes. Consumed by
`skmem::bench::run_bench` and `tests/class_bench.rs`. Run the table with
`cargo run --example class_bench`. A wrong fixture is fixed in `gen_fixtures.py`
upstream and re-copied, never edited here.
