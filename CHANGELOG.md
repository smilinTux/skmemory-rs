# Changelog

All notable changes to `skmem` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bootstrap: crate `skmem` (library plus `skmem` binary that prints the version),
  optional `python` feature (pyo3 0.24, `abi3-py39`) producing module `skmem_core`
  with a maturin mixed layout, CI (rustfmt, clippy with `-D warnings`, tests on the
  1.85 floor and stable, python feature build), secret scan, docs check, and the
  required docs. No engine code yet.
