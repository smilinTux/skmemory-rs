# skmemory-rs

[![CI](https://github.com/smilinTux/skmemory-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/smilinTux/skmemory-rs/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The Rust engine for [skmemory](https://github.com/smilinTux/skmemory), the SK
ecosystem's agent memory system. Crate `skmem`, Python module `skmem_core`.

**Status: bootstrap (0.1.0).** The crate currently exposes only its version. There is
no memory engine here yet; this repo exists so the build, test, lint and wheel
pipeline is proven before any engine code moves in.

## The plan (strangler)

1. **The engine moves to Rust.** Storage, indexing, search and retrieval, the
   performance-sensitive core of skmemory, are reimplemented here behind a small API,
   one slice at a time, with the Python implementation as the behavioural reference.
2. **The Python long tail stays Python.** Importers, the ritual, the setup wizard and
   other glue remain in `skmemory` and call into `skmem_core` through the optional
   PyO3 bindings. Nothing is rewritten that does not need to be.
3. **Strangle, do not replace.** Each slice ships behind the existing Python API and
   is switched on only when it matches the Python results on the shared fixtures.
   The Python path stays until the Rust path has earned its removal.

### What is held

The retrieval fusion work (card R-06) is **held**. It is governed by the stop rule of
the SKRSI target `skmemory-retrieval`: it does not proceed until live results hold.
Bootstrap work does not touch fusion, and nothing here claims a retrieval improvement.

## Layout

| Path | What |
| --- | --- |
| `src/lib.rs` | library crate `skmem` |
| `src/main.rs` | binary `skmem`, prints the crate version |
| `src/intentions.rs`, `src/intentions/tick.rs` | intention store, prefilter, tick (SKMEM-R-07) |
| `src/ledger.rs` | outcome ledger (SKMEM-R-07) |
| `tests/intent_parity/` | Python-generated parity vectors and the live cross-engine checks |
| `python/skmem_core/` | pure-Python half of the maturin mixed layout |
| `pyproject.toml` | maturin build, module `skmem_core` |

## Build

```
cargo test                                   # pure Rust, no Python needed
cargo run                                    # prints the version
cargo clippy --all-targets -- -D warnings
cargo build --features python                # also compile the PyO3 module

python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop                              # builds and installs skmem_core
python -c "import skmem_core; print(skmem_core.version())"
```

Rust 1.85 or newer, edition 2021. The `python` feature is off by default and uses
pyo3 0.24 with the `abi3-py39` stable ABI, so one wheel covers CPython 3.9+.

## Docs

[SOP](SOP.md), [CHANGELOG](CHANGELOG.md), [SECURITY](SECURITY.md),
[CONTRIBUTING](CONTRIBUTING.md), [AGENTS](AGENTS.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
