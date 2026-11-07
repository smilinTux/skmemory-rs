#!/usr/bin/env python3
"""Generate tests/golden/reader_parity.json from the CURRENT Python skmemory.

Hermetic: every case is synthetic. Nothing reads an agent store, a database or
the network. The expected values are the output of skmemory's own functions:

    skmemory.audience         resolve_reader, ReaderScope.allows / narrow
    skmemory.query_sanitizer  sanitize_query

The audience config is a small invented file written to a temp directory (the
real default config names real people and is never read).

Run (read-only against the skmemory checkout, no bytecode written there):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<skmemory checkout> python tests/golden/gen_reader_golden.py

The skmemory commit used is recorded in the output's ``meta``.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

from skmemory import audience as aud
from skmemory.query_sanitizer import sanitize_query

HERE = Path(__file__).resolve().parent
OUT = HERE / "reader_parity.json"

CONFIG = {
    "channels": {
        "ch-work": {"name": "Work", "context_tag": "@work:acme", "members": ["Ann", "Bob"]},
        "ch-inner": {"name": "Inner", "context_tag": "@inner:fam", "members": ["Bob", "Cy"]},
        "ch-secret": {"name": "Secret", "context_tag": "@chef-only", "members": ["Ann"]},
        "ch-empty": {"name": "Empty", "context_tag": "@chef-only", "members": []},
        "ch-nomember-known": {"name": "Ghosts", "context_tag": "@work:x", "members": ["Nobody"]},
        "ch-notag": {"name": "NoTag", "members": ["Cy"]},
        "Ch-Case": {"name": "Case", "context_tag": "@community", "members": ["Dee"]},
    },
    "people": {
        "Ann": {"trust_level": 2, "never_share": ["romantic", "revenue"]},
        "Bob": {"trust_level": 3, "never_share": ["revenue"]},
        "Cy": {"trust_level": 3, "never_share": []},
        "Dee": {"trust_level": 1, "never_share": ["intimate"]},
    },
}

READERS = [
    None, "", "  ", "@chef", "@CHEF", " @Chef-Only ", "owner", "OWNER", "@public", "@PUBLIC",
    "@community", "@work-circle", "@inner-circle", "@work:acme", "@Work:Acme", "@work:", "@inner:fam",
    "@inner:", "@work", "@unknown", "root", "*", "channel:ch-work", "channel:ch-inner",
    "channel:ch-secret", "channel:ch-empty", "channel:ch-nomember-known", "channel:ch-notag",
    "channel:Ch-Case", "channel:ch-case", "channel:CH-WORK", "channel:nope", "channel:", "channel",
    "Channel:ch-work", "Kelvin", "@work:ß",
]
ENVS = [None, "", "   ", "@public", "@community", "@inner-circle", "@chef-only", "owner",
        "@work:acme", "channel:ch-work", "channel:nope", "bogus"]
CONTEXTS = [None, "", "@public", "@community", "@work-circle", "@inner-circle", "@chef-only",
            "@work:acme", "@work:other", "@inner:fam", "@work:", "@Work:Acme", "@x"]
TAGSETS = [[], ["a"], ["romantic"], ["revenue"], ["intimate"], ["a", "revenue"], ["Romantic"]]

SANITIZE = [
    "", "   ", "\n\t", "hello", "  padded  ", "x" * 200, "x" * 201, "y" * 700,
    "What is the capital of France?",
    "You are a helpful assistant. " + "Follow the rules. " * 20 + "What did we decide about the deploy?",
    "You are an agent.\nSystem: be nice.\n" + "filler words here. " * 15 + "\nWhere is the config file?",
    "a" * 150 + ". " + "b" * 100 + "? " + "c" * 50,
    ("Lorem ipsum dolor sit amet. " * 12) + "Is this the last question? Trailing statement here.",
    ("Lorem ipsum dolor sit amet. " * 12) + "Short?",
    ("Lorem ipsum dolor sit amet. " * 12) + "Why?\nHow does the thing work in practice?",
    ("No question marks in this one. " * 12) + "Final sentence is the one we keep.",
    ("No question marks in this one. " * 12) + "Instructions: ignore everything else.",
    ("Important: do the thing. " * 15) + "Tiny.",
    ("Important: do the thing. " * 15),
    "You are a bot. " * 30,
    "word " * 120,
    "word " * 120 + "?",
    "?" * 250,
    "?" + "x" * 250,
    "x" * 250 + "?",
    "a. " * 100 + "b?",
    "a! " * 100 + "wow it works!",
    "Line one\nLine two\n" * 15 + "System: boot\nWhat now friend?",
    "You are a x\nsystem: y\n## rules\nmandatory rules\nIMPORTANT: z\n" + "pad " * 60 + "\nreal question here, right?",
    "q" * 250 + "?\n" + "z" * 5,
    ("ünïcödé wörds " * 20) + "Où est la bibliothèque municipale?",
    ("日本語のテキスト。" * 40) + "これは質問ですか?",
    "emoji \U0001F600 " * 40 + "Which \U0001F600 is it?",
    "tab\tsep\twords. " * 20 + "after emspace sentence goes here. Final one right here.",
    "A." + " " * 5 + "B" * 250,
    "x" * 190 + " " * 20 + "y" * 10,
    ("sentence one is here.  sentence two is here. " * 10) + "the end of it all",
    "z" * 499 + "\n" + "a?" ,
    "p" * 300 + "? " + "tail text with no punctuation at the end of it " * 5,
    "First. Second! Third? Fourth. " * 12,
    "x.y?z " * 60,
]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="skrs-reader-gen-") as td:
        cfg_path = Path(td) / "audience_config.json"
        cfg_path.write_text(json.dumps(CONFIG), encoding="utf-8")
        resolver = aud.AudienceResolver(cfg_path)
    readers = []
    for env, reader in itertools.product(ENVS, READERS):
        if env is None:
            os.environ.pop(aud.READER_ENV, None)
        else:
            os.environ[aud.READER_ENV] = env
        scope = aud.resolve_reader(reader, resolver)
        # one char per (context, tagset) pair, contexts outer, tagsets inner
        allows = "".join(
            "1" if scope.allows(ctx, tags) else "0" for ctx in CONTEXTS for tags in TAGSETS
        )
        readers.append(
            {
                "env": env,
                "reader": reader,
                "allowed": None if scope.allowed_tags is None else sorted(scope.allowed_tags),
                "exclusions": sorted(scope.exclusions),
                "unrestricted": scope.unrestricted,
                "allows": allows,
            }
        )
    os.environ.pop(aud.READER_ENV, None)

    sha = subprocess.run(
        ["git", "-C", str(Path(aud.__file__).resolve().parent.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    out = {
        "meta": {"skmemory_commit": sha, "generator": "tests/golden/gen_reader_golden.py"},
        "config": CONFIG,
        "contexts": [c for c in CONTEXTS],
        "tagsets": TAGSETS,
        "readers": readers,
        "sanitize": [{"in": q, "out": sanitize_query(q)} for q in SANITIZE],
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} readers={len(readers)} sanitize={len(SANITIZE)}")


if __name__ == "__main__":
    main()
