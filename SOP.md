# SOP: skmem

Standard Operating Procedure for the `skmem` crate (repo `skmemory-rs`), following the
sk-standards SK_REPO_DOC_STANDARD section layout.

## 1. Overview

`skmem` is the Rust engine for skmemory under the strangler plan: the engine moves to
Rust, the Python long tail (importers, ritual, wizard) stays Python and calls in
through the optional `python` feature (module `skmem_core`).

**Current state:** bootstrap. The library exposes `skmem::version()` and the binary
prints it. There is no engine yet.

**Held:** the retrieval fusion card (R-06) is held by the stop rule of the SKRSI target
`skmemory-retrieval` until live results hold.

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

None yet. The crate reads no environment and no files.

## 7. API

| Item | Description |
| --- | --- |
| `skmem::version()` | the crate version string |
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
