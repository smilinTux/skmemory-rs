#!/usr/bin/env bash
# Each crate here MUST fail to build, and fail for the EXPECTED reason: any other
# failure (a stale lockfile, a missing dependency) makes this script exit non-zero.
# Usage: tests/forge_check/check.sh [extra cargo args, e.g. --offline]
set -u
here=$(cd "$(dirname "$0")" && pwd)
status=0
expect() { # <crate dir> <regex that must appear in the compiler output>
  local out
  if out=$(cargo check --locked "${@:3}" --manifest-path "$here/$1/Cargo.toml" 2>&1); then
    echo "FAIL $1: it compiled, so a forging constructor is reachable"; status=1; return
  fi
  if echo "$out" | grep -Eq "$2"; then
    echo "ok   $1: refused as expected"
  else
    echo "FAIL $1: failed, but not for the expected reason ($2):"; echo "$out" | tail -25; status=1
  fi
}
expect no_feature 'error\[E0599\]: no .*named `resolve` found for struct' "$@"
expect testing_feature 'may only be enabled by skmem.s own test builds' "$@"
exit $status
