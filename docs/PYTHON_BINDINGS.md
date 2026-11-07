# Python bindings: the scoped skmem API (SKMEM-R-07B)

`skmem_core.Engine` is the surface the R-08 shim calls. It is built with the `python` feature
(maturin enables `extension-module`) and ships as one abi3 wheel. Nothing here is wired into
production: there is no Python shim in this repo.

## Scope rules

- No binding accepts, returns or names a scope object. `OwnerScope`, `ReaderScope` and `Env` are
  never exposed. A pytest asserts that no name in the module mentions scope, owner or `resolve`.
- Every call derives its scope from the PROCESS on that call:
  - reads use `ReaderScope::from_process_with_config(reader, cfg)`: the per-call `reader` can only
    narrow the `SKMEMORY_READER` ceiling, and the ceiling is resolved under
    `SKMEMORY_AUDIENCE_CONFIG`, never the `audience_config` a caller passes to `Engine(...)`;
  - owner-only calls use `OwnerScope::from_process()` and raise `PermissionError` under a ceiling.
- A hidden id reads exactly like a missing one: `recall` returns `None`, `history` returns `[]`,
  `reader_can_see` is `False`. Nothing says "forbidden".
- A process that declares a ceiling never creates the store layout (`Engine(...)` opens read-only).
- The `testing` feature (env-injecting constructors) cannot reach the wheel: it is a
  `compile_error!` with `extension-module`, and `tests/forge_check/testing_extension` proves it in CI.
- Python can still edit its own `os.environ`; that is the same trust boundary as the Python
  implementation and is not something a binding can close.

## Bindings and their scope

| Binding | Scope derivation |
|---|---|
| `Engine(memory_dir, audience_config=None, fts=True, *, agent, read_only, pg, pg_read_only)` | opens read-only under a ceiling (see "Read-only opens") |
| `search(query, limit=10, *, tags, layer, source, context_tag, record_access, reader, as_of, detail)` | reader scope (`from_process_with_config`); reinforces only for the owner |
| `search_report(...)` (same arguments) | same; returns `{"results": [...], "report": {...}}` with per-arm status |
| `recall(id, reader=None)` | reader scope; reinforces only for the owner |
| `list(layer, tags, limit, created_after, created_before, source, reader)` | reader scope |
| `history(id, reader=None)` | reader scope |
| `reader_can_see(id, reader=None)` | reader scope |
| `health(reader=None)` | reader scope; the owner gets the report, a reader only `{engine, ok, status}` |
| `snapshot(title, content, **fields)` | owner only; refuses what Python decomposes (see "Snapshot side effects") |
| `save(memory_dict, seal=True)` | owner only |
| `delete(id, reason, agent)` | owner only |
| `archive(id, sub)` | owner only |
| `reindex(force)` | owner only |
| `remind(title, due_at, interval_s, origin_memory_id, channel)` | owner only |
| `intentions(state)` | owner only |
| `intention_done(id, event)` | owner only |
| `ledger_record(skill, version, event, effect, session, host)` | owner only |
| `ledger_top(limit, host)` | owner only |
| `version()`, `__version__` | none |

Errors: `PermissionError` (owner-only under a ceiling), `ValueError` (bad `as_of`, layer, `detail`,
id, due time), `OSError` (I/O), `RuntimeError` (anything else).

## Field-name mapping to Python `MemoryStore`

Memory dicts are `Memory.to_value()`, the Rust twin of Python `Memory.model_dump()`; the keys, their
order and the nested `emotional` dict are identical (`tests/python/test_bindings.py` pins the list
and, when `skmemory` is importable, compares it to the real model).

| Python | Binding |
|---|---|
| `store.search(...) -> list[Memory]` | `search(...)` -> `list[dict]`: `model_dump()` keys plus `superseded` and `superseded_by`, exactly the rows the MCP `memory_search` builds |
| `memory_search(detail="index")` | `search(..., detail="index")`: `disclosure.index_row` keys (`id, title, date, layer, score, score_kind, est_basis, snippet, source_ref, expandable, superseded, superseded_by, est_tokens, est_tokens_full`) |
| `store.recall(id, reader)` | `recall(id, reader)` -> `dict` or `None`, with `metadata.integrity_warning` on a hash mismatch; reinforces |
| `store.list_memories(...)` | `list(...)`, newest first; `created_after` inclusive, `created_before` exclusive |
| `store.history(id, reader)` | `history(id, reader)` -> rows `{id, title, layer, created_at, valid_from, valid_to, supersedes_id, superseded_by, current, requested}` |
| `store.reader_can_see` | `reader_can_see` |
| `store.snapshot(...)` | `snapshot(title, content, **fields)`; `fields` override any `Memory` field (`layer`, `tags`, `context_tag`, `source_tier`, ...); dedup, tier, authority tier and valence as Python, then sealed |
| `store.forget` | `delete`: tombstone, flat file, index row |
| `store.reindex` | `reindex` -> `{indexed, unloadable, orphans_kept, usage_restored}` |
| `memory_remind`, `memory_intentions`, `memory_intention_done` | `remind` -> `{id, state, due_at}`; `intentions` -> `Intention.model_dump_json()` dicts; `intention_done` -> `{id, state}` or `None` |
| outcome ledger | `ledger_record` -> event id; `ledger_top` -> score dicts |

## The arm set (SKMEM-R-07C)

`search` runs the same arms as Python `MemoryStore.search_fused`, through one Rust `Engine`
(`skmem::engine`), then the same post-arm pipeline (route, vector pin, interleave or RRF,
rare-term rerank, supersession demotion, decay when configured, tier cap, procedure order).

| Arm | Backend | Notes |
|---|---|---|
| `semantic` | pgvector `PGVectorBackend.search_semantic` twin | needs `SKMEMORY_PG_DSN`; query embedded by the HTTP client (`SKMEMORY_EMBED_URL`, mxbai 1024, same request shape, token limit and failover as Python) |
| `lexical` | pg_search BM25 when Postgres is attached, else SQLite `LIKE` (`search_text`) | as Python: the pin order prefers pg BM25 when a Postgres backend is attached |
| `lexical_local` | SQLite FTS5 bm25 | always on, offline |
| `graph` | AGE entity expansion | OFF unless `SKMEMORY_GRAPH_ARM` is on, reported `disabled` otherwise |
| `temporal` | SQLite created-at window | only when the query names a time window |

- **Postgres is attached only when `SKMEMORY_PG_DSN` is set** (there is no implicit DSN), scoped to
  the agent: `agent=` or `<home>/agents/<agent>/memory`. A store whose agent cannot be derived does
  not guess one: its Postgres arms report `failed`. `pg=False` leaves them off. Connections are made
  lazily on the first search, inside the arm's worker thread and under the deadline, so a dead
  server is a per-search `failed` status that is retried, never an engine that cannot open.
- **One deadline** (`SKMEMORY_SEARCH_DEADLINE_S`, default 4 s) covers every arm, the SQLite arms
  included (a watchdog interrupts the SQLite statement when it expires and the arm is `timeout`;
  flat-file hydration afterwards is outside it). An arm still
  running at the deadline is `timeout`, never an empty result, and `report["degraded"]` is
  `"arms timed out"`. The abandoned call is left to finish and its result is dropped; while it
  runs, the next search reports that arm `timeout` at once instead of queuing behind it (Python's
  `ArmGuard`), and after a timeout the arm cools down for `SKMEMORY_ARM_COOLDOWN_S` (default 10 s):
  it is not called and reports `timeout` ("cooling down ..."); the abandoned call finishing normally
  clears it. The in-flight cap is 1 (Python's `SKMEMORY_ARM_MAX_INFLIGHT` default is 2). When nothing fused and an arm timed out the page is empty and the legacy path is not
  taken (it would call the hung backends again).
- Each arm is `ok`, `abstained` (ran, no hits), `unavailable` (no backend), `failed` (raised),
  `disabled` (switched off by config) or `timeout`. `search_report` returns them per arm with the
  ids each contributed, per-arm `hits` and `ms` (Python's `retrieval_log` shape, with `elapsed_ms`;
  a timed-out arm's `ms` is the time waited), the `route`, `rerank_applied`, `path` (`fused`, `legacy_like`, `empty`) and
  timings. It never carries the query text. Writing `retrieval.jsonl` from it stays in the shim.
- Scope: every arm receives the reader scope and filters in its own query; hydrated rows are
  re-checked; `search` drops any id the reader may not see before fusing (R-06C, R-06D, R-07B).

### Read-only opens

- A process under a `SKMEMORY_READER` ceiling opens `index.db` READ-ONLY: SQLite URI `mode=ro`
  plus `PRAGMA query_only`, no schema setup, no migration, no WAL checkpoint, no FTS creation. A
  missing `index.db` is an EMPTY IN-MEMORY index: a reader never creates a file or a directory.
  A reader MAY create `index.db-wal` and `index.db-shm` beside the database: SQLite needs them to read
  a WAL-mode file, and there is no read-only way around that without `immutable=1`, which is unsafe
  on a live index. `index.db` itself is never written; the reader needs write access to the
  directory only for those sidecars.
  Its Postgres connections are opened `default_transaction_read_only=on`.
- `read_only=True` gives an owner process the same (a live diagnostic); every write then raises
  `PermissionError`. `read_only=False` under a ceiling raises `PermissionError`.
- Reinforcement (`access_count`, `importance`, `last_accessed`) is OWNER ONLY. `recall` and
  `search(record_access=True)` bump nothing under a reader scope, whether the ceiling or a per-call
  `reader=` narrowing. **Python differs:** `MemoryStore.recall` reinforces for any reader.

### `layer` and `role` are plain `str`

Results carry `layer`, `role`, `kind`, `source_tier` as plain `str` (what `model_dump()` gives for
`use_enum_values`). Python's `MemoryLayer(str, Enum)` compares equal to the plain value and is
accepted wherever a layer is taken (`search(layer=)`, `list(layer=)`, `snapshot(layer=)`); a caller
that wants the enum wraps it: `MemoryLayer(m["layer"])`. `tests/python` pins both directions.

## Snapshot side effects (SKMEM-R-07C part 2)

`snapshot` reproduces the write-time behaviour of `MemoryStore.snapshot` that is a pure function of
the arguments and the index, each checked against the real Python store case by case
(`tests/snapshot_parity.rs`, vectors from `tests/snapshot_parity/gen_snapshot_parity.py`, 35 cases).
Every other side effect is listed here with who does it. "Shim" means the R-08 shim keeps calling the
Python store (or its backend) for it.

| Python side effect | Rust `snapshot` | Who does it |
|---|---|---|
| Content-hash dedup (`sha256(content)[:16]`; a duplicate returns the stored memory and writes nothing) | yes | Rust. The caller is the owner, so every memory is visible; Python applies the writer's `reader` here |
| `source_tier` from `source` when none is given (`tier_for_source`) | yes | Rust |
| `metadata.authority_tier` (`prepare_metadata`) | yes | Rust |
| Valence inferred from emotion labels | yes | Rust |
| Integrity seal | yes | Rust |
| Pre-write schema validation (the default hook) | yes (`Memory` validation) | Rust. Custom registered hooks: shim |
| Flat file (atomic, durable) and SQLite index row | yes | Rust |
| Document decomposition (`len >= 1200` or `decompose=True`) and content splitting | NOT done: raises `NotImplementedError`, stores nothing | Shim: call the Python snapshot |
| Write-time triage (`kind` fact, rule, preference; `supersedes_id`: parent link, `write_triage` metadata, closing the superseded head's `valid_to`) | NOT done: `supersedes_id` and `kind` are stored as given, nothing is closed or linked | Shim |
| pgvector row (embedding, upsert) | NOT done | Shim, or `skmemory sync --vector` rebuilds it from the flat files |
| Graph upsert (AGE `index_memory`) | NOT done | Shim, or `skmemory sync --graph` |
| Reweave backward pass (opt-in) | NOT done | Shim |
| SKSeed truth check (opt-in) | NOT done | Shim |
| Write-ahead log `wal/write_log.jsonl` | NOT done | Shim |
| Retrieval log (`logs/retrieval.jsonl`) | NOT done: `search_report` returns the data | Shim |
| `sync_failures` rows when vector or graph indexing fails | NOT done (there is no vector or graph write) | Shim |

A shim that wants the full Python write routes every `snapshot` to Python until the vector and graph
writers are ported; the Rust `snapshot` is correct for short content with none of those side effects
wanted, and refuses rather than diverge for the rest.

## Differences from Python (deliberate or not yet ported)

- **Temporal window parser:** `skmem::bench::time_window` returns EXACTLY Python's
  `parse_time_window(strict=True)` window or abstains; it never returns a different window.
  `tests/temporal_parity.rs` checks 5604 Python-generated queries (every phrase family: ISO dates and
  months, month plus day or year, bare years, since/before/between, relative days/weeks/months,
  last/past N, "the past month", weekdays, this/last week/month, may/march, punctuation, unicode,
  pairs): 3105 exact, 1642 abstained where Python answers, 0 wrong. It answers the relative phrases
  in Python's regex order and a lone unambiguous month name right after a time preposition. It
  abstains on an ISO-shaped date, a non-ASCII digit, any digit or a second month word near a month
  name, bare years, `may` and `march`, and the cue forms (`of june`, `june 2025`, ...). When the
  temporal arm abstains where Python answers, that arm contributes nothing the Python arm would have.
- **Legacy path:** when no arm fused and none timed out, Python tries the pgvector `search_text`
  (embedding KNN, then BM25, then ILIKE, unscoped) before SQLite. Rust goes straight to the SQLite
  `LIKE` path (scoped); the pgvector `search_text` fallback is not ported.
- **Lazy FTS backfill:** Python's `search_lexical_local` backfills missing FTS rows while searching
  (a write). The Rust search never writes; `reindex` and `fts_backfill` are owner calls.
- Links to memories the reader cannot see are REDACTED from `related_ids`, `parent_id`,
  `supersedes_id` and `metadata.superseded_by` (R-06C). Python does not do this yet.
- `snapshot`: see "Snapshot side effects" above for exactly what runs and what does not.
- `health` reports the SQLite primary only (sync drift, WAL, FTS coverage); no pg, vector or graph
  probes.
- `delete` and `archive` also remove the SQLite row; the pg derived store is untouched.
- `record_access=True` reinforces hits through `Index::bump_access` with the one-hour debounce, for
  the owner only.
