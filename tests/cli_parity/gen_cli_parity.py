#!/usr/bin/env python3
"""Generate the Python-written CLI parity vectors (SKMEM-R-10).

Usage (needs the skmemory venv; touches only a scratch dir and tests/cli_parity/):

    PYTHON=~/.skenv/bin/python python tests/cli_parity/gen_cli_parity.py <scratch-dir>

The store is the golden corpus (`tests/golden/out/corpus-*.json` plus the unicode
`@work:chiro` record, files exactly as Python saved them) with the intention and
ledger fixtures of `tests/intent_parity/`. Each case runs the real Python CLI
(`skmemory.cli`) in its own fresh copy of that store, after a `reindex`, with
SKMEMORY_READER set per case, and records stdout and the exit status. The Rust
test (`tests/cli_parity.rs`) runs `skmem` on the same store and cases and must
print the same bytes. Synthetic data only. Regenerate, never hand-edit.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = os.environ.get("PYTHON", sys.executable)
AS_OF = "2026-11-01T12:00:00+00:00"
NO_READER = None

# (name, reader ceiling, args). `list-*` cases are compared as sets of record blocks: the
# corpus shares one created_at, and Python leaves the order of ties to SQLite row order.
CASES = [
    ("search-rare-id", None, ["search", "prb-6f069c5e"]),
    ("search-rare-id-limit1", None, ["search", "prb-6f069c5e", "--limit", "1"]),
    ("search-incident", None, ["search", "inc-9b2e41"]),
    ("search-embedding-now", None, ["search", "embedding model"]),
    ("search-as-of-past", None, ["search", "embedding model", "--as-of", "2026-10-20T00:00:00+00:00"]),
    ("search-as-of-future", None, ["search", "embedding model", "--as-of", "2026-12-01"]),
    ("search-agent-default", None, ["search", "default agent", "--limit", "3"]),
    ("search-none", None, ["search", "zzzzqqqq"]),
    ("search-bad-as-of", None, ["search", "x", "--as-of", "not-a-date"]),
    ("search-private-owner", None, ["search", "medication timing"]),
    ("search-private-public-reader", "@public", ["search", "medication timing"]),
    ("search-private-reader-flag", None, ["search", "salary discussion", "--reader", "@public"]),
    ("search-rare-id-public-ceiling", "@public", ["search", "prb-6f069c5e"]),
    ("recall-exact", None, ["recall", "m-6f3b104385"]),
    ("recall-prefix", None, ["recall", "m-6f3b10"]),
    ("recall-private-owner", None, ["recall", "m-34d500c6d0"]),
    ("recall-private-public", "@public", ["recall", "m-34d500c6d0"]),
    ("recall-public-public", "@public", ["recall", "m-6f3b104385"]),
    ("recall-superseded", None, ["recall", "m-06159e9b16"]),
    ("recall-unicode-work", None, ["recall", "11111111-1111-4111-8111-111111111111"]),
    ("recall-unicode-work-public", "@public", ["recall", "11111111-1111-4111-8111-111111111111"]),
    ("recall-missing", None, ["recall", "m-doesnotexist"]),
    ("list-all", None, ["list", "--limit", "100"]),
    ("list-layer-long", None, ["list", "--layer", "long-term"]),
    ("list-layer-mid", None, ["list", "--layer", "mid-term", "--limit", "100"]),
    ("list-tags", None, ["list", "--tags", "standup"]),
    ("list-tags-two", None, ["list", "--tags", "host,skap08"]),
    ("list-none", None, ["list", "--tags", "nosuchtag"]),
    ("list-public", "@public", ["list", "--limit", "50"]),
    ("reindex", None, ["reindex"]),
    ("ledger-top", None, ["ledger", "top", "--as-of", AS_OF, "--base", "{MEM}"]),
    ("ledger-top-limit2", None, ["ledger", "top", "--as-of", AS_OF, "--limit", "2", "--base", "{MEM}"]),
]


def build_store(dst: Path) -> None:
    mem = dst / "agents" / "fx" / "memory"
    for f in sorted((ROOT / "golden" / "out").glob("*.json")):
        if not (f.name.startswith("corpus-") or f.name == "full-v2-unicode.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        tier = mem / d["layer"]
        tier.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, tier / f"{d['id']}.json")
    shutil.copytree(ROOT / "intent_parity" / "py_store" / "intentions", mem / "intentions")
    shutil.copytree(ROOT / "intent_parity" / "py_ledger" / "ledger", mem / "ledger")
    cfg = dst / "agents" / "fx" / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    # Python's `recall` prefix scan reads Path.home()/.skcapstone/agents/<agent>/memory
    # and ignores SKCAPSTONE_HOME; point that at the same store.
    os.symlink(".", dst / ".skcapstone")


def run(home: Path, reader, args):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "SKCAPSTONE_HOME": str(home),
        "SKAGENT": "fx",
        "SKCAPSTONE_AGENT": "fx",
        "SKMEMORY_AGENT": "fx",
        "SKMEMORY_EMBED_URL": "http://127.0.0.1:1/api/embed",
        "SKMEMORY_NO_VECTOR": "1",
        "PYTHONHASHSEED": "0",
    }
    if reader:
        env["SKMEMORY_READER"] = reader
    mem = str(home / "agents" / "fx" / "memory")
    args = [a.replace("{MEM}", mem) for a in args]
    code = "from skmemory.cli import cli; cli()"
    p = subprocess.run(
        [PY, "-c", code, "--no-vector", *args], env=env, capture_output=True, text=True
    )
    return p


def main() -> None:
    scratch = Path(sys.argv[1]).resolve()
    out = {}
    for name, reader, args in CASES:
        home = scratch / name
        if home.exists():
            shutil.rmtree(home)
        build_store(home)
        sync = (
            "from pathlib import Path\n"
            "from skmemory.profile_registry import sync_profile_registry\n"
            f"sync_profile_registry(Path({str(home)!r}), ['fx'], agents_base=Path({str(home / 'agents')!r}))\n"
        )
        subprocess.run(
            [PY, "-I", "-c", sync],
            check=True,
            env={"PATH": os.environ["PATH"], "HOME": str(home), "SKCAPSTONE_HOME": str(home)},
        )
        pre = run(home, None, ["reindex"])
        assert pre.returncode == 0, pre.stderr
        p = run(home, reader, args)
        out[name] = {
            "reader": reader,
            "args": args,
            "code": p.returncode,
            "stdout": p.stdout.replace(str(home), "{HOME}"),
        }
        print(f"{name}: exit {p.returncode}, {len(p.stdout)} bytes", file=sys.stderr)
    (HERE / "expected.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


main()
