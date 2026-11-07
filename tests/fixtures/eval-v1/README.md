# Conformance fixtures (copied, not edited)

Copied byte for byte from skmemory `skmemory/eval/fixtures/` at skmemory
origin/main `992c16fbbb08` (the Python repo is not reachable from CI):
`intentions-v1.jsonl`, `intention-transitions-v1.jsonl`, `prompts-v1.jsonl`,
`outcomes-v1.jsonl`, `outcomes-expected-v1.json`. `corpus-valid-to-v1.jsonl` is
the `(id, valid_to)` projection of `corpus-v1.jsonl`, which is all the
supersession sweep reads from an origin memory. A wrong fixture is fixed in
`gen_fixtures.py` upstream and re-copied, never edited here.
