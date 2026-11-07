# AGENTS.md

Rules for AI agents and humans working in this repo.

- This is the Rust engine for skmemory under the strangler plan. Read README.md.
- Scope is the card you were given. The fusion card R-06 is in progress (the hold was lifted 2026-11-07 after P1-09 closed); do not start other cards unasked.
- Never touch a real agent memory store. Tests use synthetic data and temp dirs.
- Work on a branch, open a PR, do not merge your own PR, never push to `main`.
- Commit before long verification runs. Uncommitted work is not saved.
- No secrets, private hostnames or private filesystem paths in any file.
- Do not use em or en dashes; use commas, colons, parentheses or hyphens.
- Name the agent that did the work in commit trailers, and only that agent.
- Checks: `cargo fmt --all --check`, `cargo clippy --locked --all-targets -- -D warnings`,
  `cargo test --locked --all-targets`, and with `--features python` for clippy.
