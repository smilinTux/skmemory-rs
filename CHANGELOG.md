# Changelog

All notable changes to `skmem` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Class bench (SKMEM-R-06B): `skmem::bench::run_bench` and `format_table` score `bm25_only`,
  `grep_substring`, `vector_only`, `fallback_chain`, the `full_context` oracle and the `candidate`
  (the Rust arms plus `skmem::search` over a real SQLite index) per query class on the vendored
  synthetic v1 fixtures (`tests/fixtures/eval-v1/corpus-v1.jsonl`, `queries-v1.jsonl`), with a
  hashed bag-of-concepts embedding (blake2b-8) and a fixture-checked subset of the temporal window
  parser. `cargo run --example class_bench` prints the table. `tests/class_bench.rs` asserts the
  candidate meets or beats the Python numbers recorded by `tests/golden/gen_class_bench_golden.py`
  for every class, and matches Python's ranked page on all 53 queries. `reimpl_fusion` is not ported.
- Reader hardening (SKMEM-R-06D): `audience::OwnerScope` (only `OwnerScope::from_process` builds one
  publicly, and it REFUSES with `NotOwner` when the real process `SKMEMORY_READER` declares a
  non-owner ceiling; the env-injecting `OwnerScope::resolve` and `ReaderScope::resolve` exist only
  under `cfg(test)` or the `testing` feature, enabled solely by the self dev-dependency, and the
  public `ReaderScope::from_process{,_with_config}` always apply the process ceiling, which
  `from_process_with_config` resolves under `SKMEMORY_AUDIENCE_CONFIG`, never the caller's config;
  the `testing` feature refuses to compile outside skmem's own test builds and with
  `extension-module`) now gates every raw or store-wide handle: `flat::Store::save`, `Index::index_memory`, `IntentionStore::{open,open_with}` and
  `Ledger::new` also take an `&OwnerScope` (a reader process cannot write, and `prefilter_safe`
  returns nothing under a ceiling); `Store::is_hidden_conflict_copy` is crate-private; CI now runs
  `cargo test --doc` and a must-not-compile `tests/forge_check`. Gated: `Index::{connection,fts_status,
  reindex,fts_backfill,backfill_importance,delete}`, `flat::Store::{archive,tombstone,
  is_tombstoned,delete}`, and `client_mut` on the vector, lexical and graph arms plus
  `GraphArm::cypher`. The existence oracles are scoped: `Index::count`, `bump_access` and
  `get_usage` take a `&ReaderScope`, a hidden id counts and bumps exactly like a missing one, and
  a restricted reader's count is its own, never the store total. A visible memory no longer names
  a hidden one: `ReaderScope::redact_links` drops `related_ids`, `parent_id`, `supersedes_id`
  and `metadata.superseded_by` targets the reader cannot see (an unknown target is dropped too),
  applied by `Store::{load,list}`, `Index::{get,all_rows,search_lexical}` and the vector and
  lexical pg arms. The bench takes the owner explicitly. The R-08 shim MUST set
  `SKMEMORY_AUDIENCE_CONFIG` to the packaged `audience_config.json` (documented in
  `skmem::audience`). **Breaking:** the signatures above changed.
- Reader ceiling and query sanitizer (SKMEM-R-06C): `skmem::audience` (`ReaderScope`,
  `AudienceConfig`, `resolve`: `SKMEMORY_READER` is a ceiling a per-call reader can only
  narrow, an unparseable reader is `@public`, a hidden id reads as missing) and
  `skmem::query_sanitizer::sanitize_query`, both with golden parity against Python
  (`tests/golden/gen_reader_golden.py`, 444 reader cases, 41 sanitizer cases). The filter is
  now MANDATORY: every public entry point that returns memories takes a `&ReaderScope`
  (`flat::Store::{load,list,flat_id_set}`, `index::Index::{get,all_rows,search_lexical}`,
  `pg::VectorArm::{search_semantic,search_vector}`, `pg::lexical_arm::LexicalArm::search_lexical`,
  every `pg::graph_arm::GraphArm` read, and `search::SearchInput::scope`). The pg arms push the
  ceiling into the SQL `WHERE` (and re-check each hydrated row); `search::search` drops hidden
  candidates and arm ids before fusing and sanitizes the query. **Breaking:** those signatures
  changed, `flat::Store::{find_file,load_path}` are crate-private, and `Candidate` gained
  `context_tag` and `tags`.
- Retrieval fusion (SKMEM-R-06): `skmem::fusion` (`run_arm`, `route_query` identifier vs
  natural, `reciprocal_rank_fusion`, `rank_interleave`, `pinned_rrf`, `fuse`: identifier
  route pins `lexical_local` then `lexical` at depth 3, natural route pins the vector head
  at `SKMEMORY_FUSION_VECTOR_PIN` (default 4) plus one lexical slot, interleave or RRF
  fill, per-arm cap 25), `skmem::rerank` (rare-term reranker, abstains without an
  identifier-shaped discriminating term), `skmem::search` (supersession demotion and the
  ordered pipeline), `skmem::tiers` (40 percent source-tier cap), `skmem::procedures`,
  `skmem::decay` and `skmem::env::Env`. Parity with the current Python on 577 golden cases
  (`tests/golden/gen_fusion_golden.py`, `tests/fusion_parity.rs`), including a
  `MemoryStore.search` run over fake backends. Not ported: threads and arm guard, temporal
  window parser, reader filter, query sanitizer.
- Bootstrap: crate `skmem` (library plus `skmem` binary that prints the version),
  optional `python` feature (pyo3 0.24, `abi3-py39`) producing module `skmem_core`
  with a maturin mixed layout, CI (rustfmt, clippy with `-D warnings`, tests on the
  1.85 floor and stable, python feature build), secret scan, docs check, and the
  required docs. No engine code yet.
- `skmem::model::Memory` (schema v2) and `skmem::flat::Store`: flat JSON store with
  durable atomic writes (temp, fsync, rename, directory fsync), tombstone and archive,
  byte-compatible with Python skmemory (`json.dumps(indent=2)` key order, ensure_ascii,
  float repr, unknown fields kept, lenient v2 vocabulary restored on save, Syncthing
  conflict copies skipped unless the only copy). Golden conformance fixtures are
  generated by `tests/golden/gen_golden.py`; the real-store check is opt-in
  (`SKMEM_REAL_DIR`) and read-only.
- `skmem::index`: SQLite index over the flat store, the Rust twin of Python's
  `sqlite_backend`, `fts_local` and `wal_guard` (bundled SQLite with FTS5 via rusqlite).
  Python's DDL verbatim, idempotent migrations in one immediate transaction (safe against
  a concurrent Python opener, nothing dropped), UPSERT that preserves `access_count`,
  `last_accessed` and `importance`, bm25 search with the Python filters, `bump_access`
  (0.02 step, debounce), legacy 0.5 importance recompute, reindex that snapshots and
  restores usage in one transaction, WAL bounds and throttled PASSIVE checkpoint. Vaulted
  mode indexes no plaintext and refuses FTS. Parity is diffed against the real Python
  backend (`tests/golden/index_parity.py`).
- `skmem::embed`: std-only embedding client (Python `_embed` parity): `SKMEMORY_EMBED_URL`,
  `_URLS` failover, `_MODEL`, `_TIMEOUT`, `_CONNECT_TIMEOUT`, `_VERIFY`; the unified
  `embed_text` formula (cut at 1400 chars), 400 halve-and-retry, typed
  `DimensionMismatch` / `ModelMismatch` / `Unavailable` errors. Plain `http://` only.
- `skmem::pg` (feature `pg`, off by default): pgvector search arm, `VectorArm::search_semantic`.
  HNSW cosine in a materialised CTE re-sorted by true distance, `ef_search` 200 and
  `iterative_scan relaxed_order` per session, agent scope, layer, tags AND, `context_tag`
  (one or any-of, `[]` fails closed), source and `as_of` pushed into the SQL `WHERE`, with the
  typed-column and `memory_json` fallback forms. DSN is never logged or in an error. Non-loopback hosts are refused (no TLS) unless
  `SKMEMORY_PG_ALLOW_REMOTE=1`; search settings are transaction-local (`set_config(..., true)`);
  a dead connection is re-established; `as_of` is parsed in Rust (`skmem::validity`, Python
  `normalize_as_of`) so `now`/`yesterday` are typed errors.
  Parity with Python `PGVectorBackend.search_semantic` on 46 cases, ids and distances (`tests/pg_parity`).
  Database tests are env-gated on `SKMEM_PG_TEST_DSN`; CI runs them against the public `smilintux/skmem-pg` image (pinned by digest, started with `pg_search,age` preloaded) so BM25 and AGE run too.
- `skmem::pg::lexical_arm` (SKMEM-R-05): `LexicalArm::search_lexical`, the twin of Python
  `search_lexical`. ParadeDB BM25 (`paradedb.boolean` over `content` and `title`, scored with
  `paradedb.score`) when `pg_search` and `memories_bm25` exist in a schema on the effective
  `search_path`, `ts_rank` over `tsv` otherwise; the choice is a capability probe, and a BM25
  statement error falls back for that call and is reported in `last_run()`. `as_of` is
  over-fetched 4x and filtered on parsed instants on the BM25 path, plain SQL on `ts_rank`.
  Scores are returned. An empty or all-missing `search_path` is a typed `EmptySearchPath`.
- `skmem::pg::graph_arm` (SKMEM-R-05): `GraphArm` over Apache AGE. Off by default
  (`SKMEMORY_GRAPH_ARM`): `search` reports `disabled by config` and opens no connection.
  `search_by_entity`, `search_by_entity_tokens` (whole query first, then rare tokens
  interleaved, one time budget), `currently_valid_memories`, `is_currently_valid`,
  `neighbors`, `supersedes_chain`, `get_related`, `get_lineage` over `RELATED_TO`,
  `SUPERSEDES`, `PROMOTED_FROM`, `CHUNK_OF`, `EXTENDS`, with the layer, tags AND,
  `context_tag`, source and `as_of` filters. Every value is a bound agtype parameter (the
  graph name, validated as an identifier, is the only inline text); each statement runs with
  `search_path = ag_catalog, "$user", public` set transaction-locally, so an empty path neither
  breaks it nor leaks. Temporal rules compare parsed instants (`validity::parse_ts`,
  `validity::valid_at`), not text: this fixes boundary answers the Python backend gets wrong for
  mixed UTC offsets (6 of 184 cases, flagged in `tests/pg_parity/r05_graph_expected.json`).
  Parity with the real Python backends on 88 lexical and 184 graph cases, ids and scores,
  generated by `tests/pg_parity/gen_r05_parity.py`.
- `skmem::intentions`, `skmem::intentions::tick`, `skmem::ledger` (SKMEM-R-07): the Rust twins
  of Python `intentions.py`, `intention_tick.py`, `ledger.py` and the scoring half of
  `ledger_sidecar.py`. Intentions: pydantic-exact flat files (field order, `Z` UTC, offsets and
  extra fields kept), the code-owned transition table (pending to done, terminal states inert,
  `interval_s` re-arm that skips missed intervals), the commutative merge lattice and Syncthing
  conflict-copy merge on reindex, the supersession sweep, `parse_due_at`, and the no-LLM prefilter
  (first-token index, whole-token phrase match with word-boundary guards, 16 KB head and tail scan
  with the cut token dropped, one total deadline, per-intention try-lock, fire cap of 3). The
  derived `intentions` table has Python's DDL and cell values. Locks are the same files as
  Python's (`<XDG_RUNTIME_DIR or temp>/skmemory-intent-locks-<uid>/<sha1(resolved base)[:16]>/<id>.lock`
  and `_tick.lock`, `flock(2)`), so both engines can run during a shadow run without racing; the
  libc calls are declared in-crate, no new dependency. Tick: cap per tick, stop on delivery
  failure, deliver then fire, casefolded tick host, argv with no shell and a 30 s timeout.
  Ledger: append-only per-host per-month JSONL with torn-tail repair, stable event ids, imported
  counters scored apart, first-party floors, `followed_error` neutral, harm only when explicit,
  idempotent Hermes usage import. Conformance: the Python eval fixtures
  (`tests/fixtures/eval-v1/`) plus Python-generated parity vectors
  (`tests/intent_parity/gen_intent_parity.py`: dumps, transitions, merges, prefilter and tick runs
  with resulting file bytes and index rows, ledger scores to the float). With `SKMEM_PYTHON` the
  live tests also prove the lock path and contention across engines and that each engine loads and
  re-dumps the other's files byte for byte. Prefilter timing: `tests/intent_bench.rs`.

### Fixed

- Audience trust fails closed (SKMEM-P0-20): an unknown channel member, or a person with a missing or
  malformed `trust_level`, now resolves to PUBLIC instead of trust 4 (the ported Python quirk that let a
  channel of unknown members see `@chef-only`). Reader golden regenerated against the fixed Python.

- `skmem::pg` vector arm: `hnsw.ef_search` is floored at the query limit and capped at 1000,
  and `hnsw.max_scan_tuples` (`SKMEMORY_HNSW_MAX_SCAN_TUPLES`, default 10000) is set with the
  iterative scan, matching skmemory PRs 147 and 148.

- `index::word_chars` (FTS query sanitiser) now follows Python `\w` through a generated table
  instead of `char::is_alphanumeric`, which accepted combining marks and circled letters.
