#!/usr/bin/env python3
"""Python side of the opt-in real-store parity run (reads only, prints counts).

argv: <file-list> <rust-out-dir>. The list holds one source path per line (the
sampled real files); Rust wrote its re-serialisation of line N to
``<rust-out-dir>/N.json`` (absent when Rust could not load that file). For each
file this loads with Python's ``Memory.from_stored_json`` and dumps like
``FileBackend.save``, then compares. Never writes next to the sources and never
prints a path, id or content.
"""
import collections
import json
import sys
from pathlib import Path

from skmemory.models import Memory

lines = Path(sys.argv[1]).read_text().splitlines()
out_dir = Path(sys.argv[2])
c = collections.Counter()
diff_keys = collections.Counter()
for n, line in enumerate(lines):
    raw = Path(line).read_bytes()
    rust_path = out_dir / f"{n}.json"
    rust = rust_path.read_bytes() if rust_path.exists() else None
    try:
        py = json.dumps(Memory.from_stored_json(raw).model_dump(), indent=2, default=str).encode()
    except Exception:
        py = None
    c["sampled"] += 1
    c["python_loads" if py is not None else "python_rejects"] += 1
    c["rust_loads" if rust is not None else "rust_rejects"] += 1
    if py is not None and rust is not None:
        c["both_load"] += 1
        if rust == py:
            c["rust_bytes==python_bytes"] += 1
        else:
            c["rust_bytes!=python_bytes"] += 1
            # A file with no updated_at is stamped with now() by BOTH sides, so
            # the two timestamps legitimately differ; mask it and compare again.
            def mask(b):
                return [x if not x.lstrip().startswith('"updated_at"') else "" for x in b.decode().splitlines()]
            if "updated_at" not in json.loads(raw) and mask(rust) == mask(py):
                c["rust!=python only by now()-stamped missing updated_at"] += 1
            a, b = rust.decode().splitlines(), py.decode().splitlines()
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            diff_keys[a[k].split(":")[0].strip()[:30] if k is not None else "length"] += 1
        if py == raw:
            c["python_dump==raw_file"] += 1
        if rust == raw:
            c["rust_dump==raw_file"] += 1
    elif py is None and rust is None:
        c["both_reject"] += 1
    elif py is not None:
        c["RUST_REJECTS_PYTHON_LOADS"] += 1
    else:
        c["RUST_LOADS_PYTHON_REJECTS"] += 1
for k in sorted(c):
    print(f"PARITY {k}={c[k]}")
for k, v in diff_keys.items():
    print(f"PARITY first-diff-key[{k}]={v}")
