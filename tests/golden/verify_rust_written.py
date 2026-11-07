#!/usr/bin/env python3
"""Check that Python loads what Rust wrote and sees the same record.

argv: <rust memory dir> <golden out dir> <manifest.json>. Uses the real
FileBackend.load, then compares model_dump() with Python's own dump of the
golden out/ file, and the raw bytes of each file Rust wrote.
"""
import json
import sys
from pathlib import Path

from skmemory.backends.file_backend import FileBackend
from skmemory.models import Memory

root, out_dir, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cases = json.loads(manifest.read_text())["cases"]
fb = FileBackend(str(root))
bad = 0
for name in cases:
    want_bytes = (out_dir / f"{name}.json").read_bytes()
    want = Memory.from_stored_json(want_bytes)
    got = fb.load(want.id)
    if got is None:
        print(f"{name}: Python could not load Rust output")
        bad += 1
        continue
    if got.model_dump() != want.model_dump():
        print(f"{name}: model_dump differs")
        bad += 1
    written = next(root.glob(f"*/{want.id}.json")).read_bytes()
    if written != want_bytes:
        print(f"{name}: bytes differ from Python's own save")
        bad += 1
print(f"checked {len(cases)} cases, {bad} problems")
sys.exit(1 if bad else 0)
