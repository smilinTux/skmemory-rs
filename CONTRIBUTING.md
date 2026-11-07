# Contributing

Read [SOP.md](SOP.md) first. The short version:

1. Branch from `main`, one branch per change, and open a PR. Do not push to `main`.
2. Use conventional-commit titles (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`).
3. Before pushing, run:
   ```
   cargo fmt --all
   cargo clippy --locked --all-targets -- -D warnings
   cargo clippy --locked --all-targets --features python -- -D warnings
   cargo test --locked --all-targets
   ```
4. A change to `src/` or `pyproject.toml` must also update `CHANGELOG.md` (the docs
   check enforces this).
5. Tests are hermetic: no network, no database, no real agent store, no real memory
   data.
6. Commit as soon as the code and a fast check pass, then run the long checks.
7. No secrets and no private hostnames or paths, anywhere.

## Scope

This repo holds the Rust engine only. Importers, the ritual and the wizard stay in
the Python `skmemory` repo. The retrieval fusion card (R-06) is held, see README.
