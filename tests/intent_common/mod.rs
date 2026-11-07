//! Helpers shared by the intention and ledger integration tests.
#![allow(dead_code)]

use rusqlite::{types::ValueRef, Connection};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

pub fn parity() -> PathBuf {
    root().join("tests/intent_parity")
}

pub fn fixtures() -> PathBuf {
    root().join("tests/fixtures/eval-v1")
}

static N: AtomicU64 = AtomicU64::new(0);

/// A fresh scratch directory (under `SKMEM_TEST_TMP` when set).
pub fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var_os("SKMEM_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!(
        "skmem-intent-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

pub fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for e in fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to);
        } else {
            fs::copy(e.path(), to).unwrap();
        }
    }
}

pub fn jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

pub fn expected() -> Value {
    serde_json::from_str(&fs::read_to_string(parity().join("expected.json")).unwrap()).unwrap()
}

fn cell(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
        ValueRef::Blob(_) => json!("<blob>"),
    }
}

pub fn table_rows(base: &Path, sql: &str) -> Vec<Value> {
    let c = Connection::open(base.join("index.db")).unwrap();
    let mut st = c.prepare(sql).unwrap();
    let n = st.column_count();
    st.query_map([], |r| {
        Ok(Value::Array(
            (0..n).map(|i| cell(r.get_ref(i).unwrap())).collect(),
        ))
    })
    .unwrap()
    .map(Result::unwrap)
    .collect()
}

/// Flat files and `intentions` rows of a store.
pub struct Snap {
    pub files: BTreeMap<String, String>,
    pub rows: Vec<Value>,
}

pub fn snapshot(base: &Path) -> Snap {
    let mut files = BTreeMap::new();
    for e in fs::read_dir(base.join("intentions")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "json") {
            files.insert(
                p.file_name().unwrap().to_string_lossy().into_owned(),
                fs::read_to_string(&p).unwrap(),
            );
        }
    }
    Snap {
        files,
        rows: table_rows(base, "SELECT * FROM intentions ORDER BY id"),
    }
}

/// Only what differs from the pristine snapshot, in the shape the generator writes.
pub fn delta(s: &Snap, p: &Snap) -> Value {
    let mut files = Map::new();
    for (k, v) in &s.files {
        if p.files.get(k) != Some(v) {
            files.insert(k.clone(), json!(v));
        }
    }
    let deleted: Vec<&String> = p.files.keys().filter(|k| !s.files.contains_key(*k)).collect();
    let rows: Vec<&Value> = s.rows.iter().filter(|r| !p.rows.contains(r)).collect();
    let ids: Vec<&Value> = s.rows.iter().map(|r| &r[0]).collect();
    json!({"files": files, "deleted": deleted, "rows": rows, "row_ids": ids})
}
