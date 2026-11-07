//! SKMEM-R-07C part 2: `Engine::snapshot` against Python `MemoryStore.snapshot`.
//!
//! `tests/snapshot_parity/expected.json` is written by `gen_snapshot_parity.py`
//! from the real Python store over a throwaway SQLite primary. Every case is
//! replayed here and must give the same `source_tier`, valence, metadata (key
//! order included) and dedup outcome; the cases Python decomposes must be refused,
//! never stored whole.

use serde_json::{json, Map, Value};
use skmem::audience::OwnerScope;
use skmem::engine::{Arms, Engine, EngineOptions};
use skmem::env::Env;
use skmem::snapshot::SnapshotError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn tmp(tag: &str) -> PathBuf {
    let base = std::env::var_os("SKMEM_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("skmem-r07c2-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn expected() -> Vec<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshot_parity/expected.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn engine(tag: &str) -> Engine {
    Engine::open(
        tmp(tag).join("memory"),
        EngineOptions {
            read_only: Some(false),
            ..EngineOptions::new(Arms::none())
        },
    )
    .unwrap()
}

fn owner() -> OwnerScope {
    OwnerScope::resolve(&Env::empty()).unwrap()
}

/// The Memory-field overrides a Python snapshot call with these kwargs implies.
fn fields(case: &Value) -> Map<String, Value> {
    let mut f = Map::new();
    for k in [
        "source",
        "tags",
        "source_ref",
        "metadata",
        "decompose",
        "layer",
        "source_tier",
        "emotional",
    ] {
        if let Some(v) = case.get(k) {
            f.insert(k.to_string(), v.clone());
        }
    }
    f
}

#[test]
fn snapshot_matches_python_case_by_case() {
    let e = engine("parity");
    let own = owner();
    let mut ids: HashMap<String, String> = HashMap::new();
    let mut checked = 0;
    for row in expected() {
        let case = &row["case"];
        let name = case["name"].as_str().unwrap();
        let res = e.snapshot(
            case["title"].as_str().unwrap(),
            case["content"].as_str().unwrap(),
            fields(case),
            &own,
        );
        if row["decomposed"].as_bool().unwrap() {
            assert!(
                matches!(res, Err(SnapshotError::Unsupported(_))),
                "{name}: Python decomposes this, Rust must refuse"
            );
            checked += 1;
            continue;
        }
        let snap = res.unwrap_or_else(|err| panic!("{name}: {err}"));
        let m = &snap.memory;
        match row["same_as"].as_str() {
            Some(first) => {
                assert!(!snap.created, "{name}: Python deduplicated");
                assert_eq!(m.id, ids[first], "{name}: dedup returns the first memory");
            }
            None => {
                assert!(snap.created, "{name}: Python wrote a new memory");
                ids.insert(name.to_string(), m.id.clone());
            }
        }
        assert_eq!(
            m.source_tier,
            row["source_tier"].as_str().unwrap(),
            "{name}: source_tier"
        );
        assert_eq!(
            m.emotional.valence,
            row["valence"].as_f64().unwrap(),
            "{name}: valence"
        );
        assert_eq!(
            m.emotional.intensity,
            row["intensity"].as_f64().unwrap(),
            "{name}"
        );
        let got = Value::Object(m.metadata.clone());
        assert_eq!(got, row["metadata"], "{name}: metadata");
        let keys: Vec<&String> = m.metadata.keys().collect();
        let want: Vec<&str> = row["metadata_keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap())
            .collect();
        assert_eq!(keys, want, "{name}: metadata key order");
        assert_eq!(m.layer.as_str(), row["layer"].as_str().unwrap(), "{name}");
        assert_eq!(json!(m.tags), row["tags"], "{name}");
        assert_eq!(m.source, row["source"].as_str().unwrap(), "{name}");
        assert_eq!(m.title, row["title"].as_str().unwrap(), "{name}");
        assert!(
            m.verify_integrity() && !m.integrity_hash.is_empty(),
            "{name}: sealed"
        );
        checked += 1;
    }
    assert_eq!(checked, expected().len());
}

#[test]
fn a_refused_snapshot_stores_nothing() {
    let e = engine("refused");
    let own = owner();
    let long = "river delta ".repeat(200);
    let r = e.snapshot("t", &long, Map::new(), &own);
    assert!(matches!(r, Err(SnapshotError::Unsupported(_))));
    assert_eq!(e.store().flat_id_set(own.reader_scope()).len(), 0);
    assert_eq!(e.index().count(own.reader_scope()).unwrap(), 0);
}

#[test]
fn a_duplicate_writes_nothing_and_returns_the_stored_memory() {
    let e = engine("dup");
    let own = owner();
    let a = e.snapshot("one", "same words", Map::new(), &own).unwrap();
    let b = e.snapshot("two", "same words", Map::new(), &own).unwrap();
    assert!(a.created && !b.created);
    assert_eq!(a.memory.id, b.memory.id);
    assert_eq!(
        b.memory.title, "one",
        "the existing memory, not the new title"
    );
    assert_eq!(e.store().flat_id_set(own.reader_scope()).len(), 1);
}
