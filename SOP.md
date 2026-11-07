# SOP: skmem

Standard Operating Procedure for the `skmem` crate (repo `skmemory-rs`), following the
sk-standards SK_REPO_DOC_STANDARD section layout.

## 1. Overview

`skmem` is the Rust engine for skmemory under the strangler plan: the engine moves to
Rust, the Python long tail (importers, ritual, wizard) stays Python and calls in
through the optional `python` feature (module `skmem_core`).

**Current state:** storage, index, embed client, pg arms, intentions and ledger, and the
retrieval fusion pipeline are in the library.

**Retrieval fusion (R-06), in progress:** the post-arm half of `MemoryStore.search` is
ported and held to exact parity with Python: `skmem::fusion` (arm classification, route
query, RRF, rank interleave, pinned RRF, `fuse`), `skmem::rerank` (rare-term reranker),
`skmem::search::demote_superseded` and the ordered `skmem::search::search` pipeline
(fuse, rerank, demote, decay, demote, tier cap, procedure order), `skmem::tiers`,
`skmem::procedures` and `skmem::decay`. Not ported: the thread pool and arm guard (the
caller runs the arms), the temporal window parser (its outcome enters `fuse` as an arm),
the reader filter, the query sanitizer, access reinforcement, the legacy fallback chain
(`search` returns `None` where Python falls back), and `run_bench()` (the class-bench
corpus and queries fixtures are not vendored here; only the `corpus-valid-to`
projection is). Python reads `os.environ`; Rust takes an explicit `skmem::env::Env`.

**Out of scope here:** importers, the ritual, the wizard, and any real data.

## 2. Architecture

- `src/lib.rs`: library `skmem` (crate types `cdylib` and `rlib`).
- `src/main.rs`: binary `skmem`, prints the crate version.
- `python` feature (off by default): pyo3 0.24 with `abi3-py39`, compiled into the
  `_skmem_core` extension.
- `python/skmem_core/`: pure-Python half of the maturin mixed layout, re-exports the
  extension as `skmem_core`.

## 3. Build

Rust 1.85 or newer, edition 2021.

```
cargo build
cargo build --features python
maturin develop        # in a venv, installs skmem_core
```

## 4. Test

```
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --features python -- -D warnings
cargo fmt --all --check
```

CI runs the same checks, plus tests on the 1.85.0 floor and a maturin wheel build with
an import smoke test. Tests are hermetic.

## 5. Release

Not published. `publish = false` until an engine exists. The version of record is
`[package] version` in `Cargo.toml`, which maturin reads for the wheel. Add a dated
CHANGELOG entry with every version bump.

## 6. Configuration

The retrieval stages read these through an explicit `Env` (never the process
environment implicitly): `SKMEMORY_FUSION_STRATEGY`, `_FUSION_PIN`, `_FUSION_VECTOR_PIN`,
`_FUSION_ROUTE`, `_FUSION_FILL`, `SKMEMORY_SEARCH_RERANK`, `_SEARCH_RERANK_DIVISOR`,
`SKMEMORY_TIER_CAP`, `SKMEMORY_DECAY`, plus the pg arm's `SKMEMORY_HNSW_EF_SEARCH`,
`_HNSW_ITERATIVE_SCAN` and `_HNSW_MAX_SCAN_TUPLES`.

## 7. API

| Item | Description |
| --- | --- |
| `skmem::version()` | the crate version string |
| `skmem::fusion::{run_arm, route_query, fuse, pinned_rrf, rank_interleave, reciprocal_rank_fusion}` | arm classification, routing and the mergers |
| `skmem::rerank::Reranker` | rare-term reranker (`SKMEMORY_SEARCH_RERANK`, `_DIVISOR`) |
| `skmem::search::{search, demote_superseded, Candidate}` | the ordered post-arm pipeline |
| `skmem::tiers::apply_tier_cap` | source-tier occupancy cap (`SKMEMORY_TIER_CAP`) |
| `skmem::procedures::order_procedures` | failure procedures never above successes for a how-to |
| `skmem::decay::{rerank, decay_score, configured_channel}` | decay channels (`SKMEMORY_DECAY`) |
| `skmem_core.version()` | same, from Python |
| `skmem_core.__version__` | same, as an attribute |

## 8. Troubleshooting

- `maturin develop` fails with "Couldn't find a virtualenv": create and activate one.
- `import skmem_core` fails: confirm the wheel was built with the `extension-module`
  feature (the pyproject sets it) and installed into the active venv.
- docs-check fails tier 2: a change to `src/` or `pyproject.toml` needs a CHANGELOG
  entry in the same PR.

## 9. Version reference

- **Source of truth:** `[package] version` in `Cargo.toml`.
- **Rust floor:** `rust-version = "1.85"`, `edition = "2021"`.
- **Maturity:** pre-alpha, bootstrap only. No maturity tier above T0 is claimed.

<!-- docs-evidence
verified: 2026-11-07
checks:
  - name: package name, lib name and Rust floor match the docs (SOP 9)
    run: grep -qE '^name = "skmem"$' Cargo.toml && grep -qE '^rust-version = "1\.85"$' Cargo.toml && grep -qE '^edition = "2021"$' Cargo.toml
  - name: python feature is optional and pyo3 is 0.24 abi3-py39 (SOP 2)
    run: grep -qE '^python = \["dep:pyo3"\]$' Cargo.toml && grep -qE 'pyo3 = \{ version = "0\.24", features = \["abi3-py39"\], optional = true \}' Cargo.toml
  - name: Python module is skmem_core in a maturin mixed layout (SOP 2)
    run: grep -qE '^module-name = "skmem_core\._skmem_core"$' pyproject.toml && grep -qE '^python-source = "python"$' pyproject.toml && test -f python/skmem_core/__init__.py
  - name: entry points named in SOP 2 exist
    run: test -f src/lib.rs && test -f src/main.rs
-->
