# skmem-mcp: the MCP server on the Rust engine

`skmem-mcp` is an MCP server (stdio, newline-delimited JSON-RPC 2.0) over the same
`Engine` the Python shim uses. Run it as `skmem-mcp [--memory-dir DIR]`. The store is
`DIR`, else `$SKMEMORY_HOME/memory`, else `<agents>/<agent>/memory` where the agent comes
from `SKAGENT`, `SKCAPSTONE_AGENT`, `SKMEMORY_AGENT` (all that are set must agree) or
`SK_DEFAULT_AGENT`. It never guesses an agent. Postgres arms attach with the `pg` feature
and `SKMEMORY_PG_DSN`, exactly as in the Python bindings.

## Tools

Served (names, descriptions and input schemas equal the Python server's, checked against
`tests/fixtures/mcp/python-tools.json`):

`memory_store`, `memory_search`, `memory_recall`, `memory_history`, `memory_list`,
`memory_remind`, `memory_intentions`, `memory_intention_done`.

Left to the Python server (not listed by `tools/list`; a call answers "not served by
skmem-mcp"): `memory_context` and `memory_health` (their result shapes are built from the
Python store, not the engine), `memory_expand`, `memory_check_duplicate`, `memory_graph`,
`memory_auto_context`, `memory_extract`, `memory_save_session`, `memory_pre_compact`
(no engine counterpart), and the owner-only maintenance tools `memory_forget`,
`memory_promote`, `memory_consolidate`, `memory_export`, `memory_import`, `memory_verify`,
`memory_audit`, `memory_synthesize_daily`, `memory_synthesize_dreams`.

## Reader policy

Same as Python. `SKMEMORY_READER` is a ceiling and a per-call `reader` can only narrow it.
Owner-only tools (including the ones left to Python) are refused to a declared reader with
`<tool> is owner-only and cannot be used by a declared reader`. A hidden id reads as
`Memory not found: <id>`. Links to hidden memories are dropped from returned rows.

## Known differences

- `memory_store` under a process-wide `SKMEMORY_READER` ceiling is refused here (the engine's
  writes need the owner); Python stores it with tier `self-generated`.
- Content of 1200+ characters is not decomposed (the engine does not); the call errors and
  stores nothing, so use the Python server for it.
- A redacted observation (some evidence hidden from the reader) is cut back to its first
  paragraph; Python keeps the visible quote lines.

## Regenerating the fixture

`PYTHONPATH=<skmemory checkout> python scripts/gen_mcp_tool_fixture.py` rewrites the fixture
and `src/mcp/schemas.rs` from the Python package. The parity test fails if they drift.
