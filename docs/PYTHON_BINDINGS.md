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
| `Engine(memory_dir, audience_config=None, fts=True)` | opens read-only under a ceiling |
| `search(query, limit=10, *, tags, layer, source, context_tag, record_access, reader, as_of, detail)` | reader scope (`from_process_with_config`) |
| `recall(id, reader=None)` | reader scope |
| `list(layer, tags, limit, created_after, created_before, source, reader)` | reader scope |
| `history(id, reader=None)` | reader scope |
| `reader_can_see(id, reader=None)` | reader scope |
| `health(reader=None)` | reader scope; the owner gets the report, a reader only `{engine, ok, status}` |
| `snapshot(title, content, **fields)` | owner only |
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
| `store.snapshot(...)` | `snapshot(title, content, **fields)`; `fields` override any `Memory` field (`layer`, `tags`, `context_tag`, `source_tier`, ...) and the memory is sealed |
| `store.forget` | `delete`: tombstone, flat file, index row |
| `store.reindex` | `reindex` -> `{indexed, unloadable, orphans_kept, usage_restored}` |
| `memory_remind`, `memory_intentions`, `memory_intention_done` | `remind` -> `{id, state, due_at}`; `intentions` -> `Intention.model_dump_json()` dicts; `intention_done` -> `{id, state}` or `None` |
| outcome ledger | `ledger_record` -> event id; `ledger_top` -> score dicts |

## Differences from Python (deliberate or not yet ported)

- Search runs the `lexical_local` arm (SQLite FTS5) through `skmem::search` (fusion, rerank,
  supersession demotion, decay when configured, tier cap, procedure order). The `semantic`,
  `lexical` (pg), `temporal` arms are reported unattached and the graph arm disabled; the pg feature
  is not in the wheel. Callers that need vectors keep the Python path until R-08 decides.
- Links to memories the reader cannot see are REDACTED from `related_ids`, `parent_id`,
  `supersedes_id` and `metadata.superseded_by` (R-06C). Python does not do this yet.
- `snapshot` does not run Python's hooks, reweave, embedding or graph indexing.
- `health` reports the SQLite primary only (sync drift, WAL, FTS coverage); no pg, vector or graph
  probes.
- `delete` and `archive` also remove the SQLite row; the pg derived store is untouched.
- `record_access=True` reinforces hits through `Index::bump_access` with the one-hour debounce.
