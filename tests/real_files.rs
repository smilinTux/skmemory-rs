//! Read-only byte-compat check against a real agent store.
//!
//! Off by default and never run in CI: set `SKMEM_REAL_DIR` to a
//! `<agent>/memory` directory (optionally `SKMEM_REAL_SAMPLE`, default 1000).
//! Files are read, parsed and re-serialised in memory and compared with the
//! original bytes. NOTHING is written there. Only counts are printed, never
//! ids or content.

use serde_json::Value;
use skmem::model::{Layer, Memory};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const V2_KEYS: &[&str] = &[
    "kind",
    "outcome",
    "event_at",
    "valid_from",
    "valid_to",
    "supersedes_id",
    "source_tier",
    "schema_version",
];

fn sample(root: &Path, want: usize) -> Vec<PathBuf> {
    let mut per: Vec<Vec<PathBuf>> = Vec::new();
    for l in Layer::ALL {
        let mut v: Vec<PathBuf> = fs::read_dir(root.join(l.as_str()))
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| {
                        let n = p.file_name().unwrap().to_string_lossy();
                        n.ends_with(".json") && !n.starts_with('.')
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        per.push(v);
    }
    let total: usize = per.iter().map(Vec::len).sum();
    let mut out = Vec::new();
    for v in per {
        // Proportional share per layer, evenly spaced through the sorted names.
        let take = ((v.len() * want).div_ceil(total.max(1))).min(v.len());
        for i in 0..take {
            out.push(v[i * v.len() / take.max(1)].clone());
        }
    }
    out.truncate(want);
    out
}

/// Bucket a mismatch by what differs, without exposing content.
fn categorise(orig: &[u8], rewritten: &str, parsed: &Value) -> &'static str {
    let Ok(text) = std::str::from_utf8(orig) else {
        return "not utf-8";
    };
    if text.trim_end() == rewritten {
        return "trailing whitespace/newline only";
    }
    let Ok(Value::Object(o)) = serde_json::from_str::<Value>(text) else {
        return "unparseable";
    };
    let Value::Object(r) = parsed else {
        return "other";
    };
    if !o.contains_key("schema_version") {
        return "v1 file (no schema_version; v2 keys added on dump)";
    }
    let ok: Vec<&String> = o.keys().collect();
    let rk: Vec<&String> = r.keys().collect();
    if ok != rk {
        let mut a = ok.clone();
        let mut b = rk.clone();
        a.sort();
        b.sort();
        return if a == b {
            "key order differs"
        } else {
            "key set differs (missing declared keys)"
        };
    }
    if !text.contains("\n  \"") {
        return "different indentation/separators";
    }
    if text.bytes().any(|b| b >= 0x80) {
        return "raw non-ASCII (ensure_ascii=False writer)";
    }
    let o2: Value = serde_json::from_str(text).unwrap();
    if &o2 != parsed {
        return "values differ (normalisation: float/str/strip)";
    }
    "formatting differs, same values"
}

#[test]
fn real_files_roundtrip() {
    let Ok(dir) = std::env::var("SKMEM_REAL_DIR") else {
        eprintln!("SKMEM_REAL_DIR unset: skipping real-file run");
        return;
    };
    let want: usize = std::env::var("SKMEM_REAL_SAMPLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let files = sample(Path::new(&dir), want);
    // Scratch directory for Rust's re-serialisations (never inside the store).
    let out_dir = PathBuf::from(
        std::env::var("SKMEM_REAL_OUT").expect("set SKMEM_REAL_OUT to a scratch directory"),
    );
    assert!(
        !out_dir.starts_with(&dir),
        "SKMEM_REAL_OUT must not be inside the store"
    );
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir).unwrap();
    let list_path = out_dir.join("files.txt");
    fs::write(
        &list_path,
        files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let out_dir = out_dir.join("rust");
    fs::create_dir_all(&out_dir).unwrap();
    let (mut identical, mut load_fail, mut v1_format_ok, mut v1_total) =
        (0usize, 0, 0usize, 0usize);
    let mut buckets: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut by_layer: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (idx, p) in files.iter().enumerate() {
        let layer = p
            .parent()
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let e = by_layer.entry(layer).or_default();
        e.0 += 1;
        let raw = fs::read(p).unwrap();
        match Memory::from_stored_json(&raw) {
            Err(_) => {
                load_fail += 1;
                *buckets
                    .entry("load failed in Rust (see PARITY lines for Python agreement)")
                    .or_default() += 1;
            }
            Ok(m) => {
                let out = m.to_json();
                fs::write(out_dir.join(format!("{idx}.json")), &out).unwrap();
                // A v1 file lacks the v2 keys Python adds on dump. Drop them from
                // our output and the remaining bytes must equal the original.
                if !String::from_utf8_lossy(&raw).contains("\"schema_version\"") {
                    v1_total += 1;
                    let mut v = m.to_value();
                    if let Value::Object(o) = &mut v {
                        for k in V2_KEYS {
                            o.shift_remove(*k);
                        }
                    }
                    if skmem::pyjson::dumps_indent2(&v).as_bytes() == raw.as_slice() {
                        v1_format_ok += 1;
                    }
                }
                if out.as_bytes() == raw.as_slice() {
                    identical += 1;
                    e.1 += 1;
                } else {
                    *buckets
                        .entry(categorise(&raw, &out, &m.to_value()))
                        .or_default() += 1;
                }
            }
        }
    }
    println!("REAL v1 files={v1_total}, byte-identical once v2 keys are dropped={v1_format_ok}");
    println!(
        "REAL sampled={} identical={} load_fail={}",
        files.len(),
        identical,
        load_fail
    );
    for (l, (n, i)) in &by_layer {
        println!("REAL layer {l}: {n} sampled, {i} identical");
    }
    for (k, v) in &buckets {
        println!("REAL mismatch [{k}]: {v}");
    }

    match std::env::var("SKMEM_PYTHON") {
        Ok(py) => {
            let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/real_parity.py");
            let out = std::process::Command::new(py)
                .arg(script)
                .arg(&list_path)
                .arg(&out_dir)
                .output()
                .expect("run python");
            print!("{}", String::from_utf8_lossy(&out.stdout));
            assert!(out.status.success(), "python parity script failed");
        }
        Err(_) => println!("PARITY not computed: set SKMEM_PYTHON (needs skmemory importable)"),
    }
}
