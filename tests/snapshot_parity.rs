//! SKMEM-R-07C part 2: `Engine::snapshot` against Python `MemoryStore.snapshot`.
//!
//! `tests/snapshot_parity/expected.json` is written by `gen_snapshot_parity.py`
//! from the real Python store over a throwaway SQLite primary. Every case is
//! replayed here and must give the same `source_tier`, valence, metadata (key
//! order included) and dedup outcome; the cases Python decomposes must be refused,
//! never stored whole.

use serde_json::{json, Map, Value};
use skmem::audience::{AudienceConfig, OwnerScope, ReaderScope};
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
            own.reader_scope(),
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
    let r = e.snapshot("t", &long, Map::new(), &own, own.reader_scope());
    assert!(matches!(r, Err(SnapshotError::Unsupported(_))));
    assert_eq!(e.store().flat_id_set(own.reader_scope()).len(), 0);
    assert_eq!(e.index().count(own.reader_scope()).unwrap(), 0);
}

#[test]
fn a_duplicate_writes_nothing_and_returns_the_stored_memory() {
    let e = engine("dup");
    let own = owner();
    let a = e
        .snapshot("one", "same words", Map::new(), &own, own.reader_scope())
        .unwrap();
    let b = e
        .snapshot("two", "same words", Map::new(), &own, own.reader_scope())
        .unwrap();
    assert!(a.created && !b.created);
    assert_eq!(a.memory.id, b.memory.id);
    assert_eq!(
        b.memory.title, "one",
        "the existing memory, not the new title"
    );
    assert_eq!(e.store().flat_id_set(own.reader_scope()).len(), 1);
}

fn public() -> ReaderScope {
    ReaderScope::resolve(Some("@public"), &Env::empty(), &AudienceConfig::empty())
}

#[test]
fn dedup_only_matches_what_the_writers_reader_may_see() {
    let e = engine("dedup-reader");
    let own = owner();
    let mut priv_fields = Map::new();
    priv_fields.insert("context_tag".into(), json!("@chef-only"));
    let secret = e
        .snapshot(
            "secret",
            "private words only",
            priv_fields,
            &own,
            own.reader_scope(),
        )
        .unwrap();
    assert!(secret.created);
    // A public writer with the same content: the hit is invisible to it, so a fresh
    // memory is written and the private one is neither returned nor confirmed.
    let mut f = Map::new();
    f.insert("context_tag".into(), json!("@public"));
    f.insert("reader".into(), json!("@public"));
    let fresh = e
        .snapshot("mine", "private words only", f, &own, &public())
        .unwrap();
    assert!(
        fresh.created,
        "a hit on an invisible memory writes a fresh one"
    );
    assert_ne!(fresh.memory.id, secret.memory.id);
    assert_eq!(fresh.memory.title, "mine");
    assert!(
        !fresh.memory.extra.contains_key("reader"),
        "reader is never a stored field"
    );
    // The stored file has no reader key either.
    let file = e.store().path_for(&fresh.memory).unwrap();
    assert!(!std::fs::read_to_string(file)
        .unwrap()
        .contains("\"reader\""));
    // The owner (no reader) still dedups to the first row of that hash.
    let again = e
        .snapshot(
            "x",
            "private words only",
            Map::new(),
            &own,
            own.reader_scope(),
        )
        .unwrap();
    assert!(!again.created);
    // And a reader that CAN see the hit dedups to it.
    let mut pubf = Map::new();
    pubf.insert("context_tag".into(), json!("@public"));
    let shared = e
        .snapshot("shared", "open words", pubf, &own, own.reader_scope())
        .unwrap();
    let mut g = Map::new();
    g.insert("reader".into(), json!("@public"));
    let seen = e.snapshot("y", "open words", g, &own, &public()).unwrap();
    assert!(!seen.created);
    assert_eq!(seen.memory.id, shared.memory.id);
}

#[test]
fn decompose_truthiness_matches_python() {
    let e = engine("decompose-truthy");
    let own = owner();
    for (i, v) in [json!(1), json!("yes"), json!([0]), json!(true)]
        .into_iter()
        .enumerate()
    {
        let mut f = Map::new();
        f.insert("decompose".into(), v);
        let r = e.snapshot("t", &format!("short {i}"), f, &own, own.reader_scope());
        assert!(
            matches!(r, Err(SnapshotError::Unsupported(_))),
            "truthy decompose #{i}"
        );
    }
    for (i, v) in [json!(0), json!(""), json!([]), json!(null), json!(false)]
        .into_iter()
        .enumerate()
    {
        let mut f = Map::new();
        f.insert("decompose".into(), v);
        let r = e.snapshot("t", &format!("falsy {i}"), f, &own, own.reader_scope());
        assert!(r.is_ok(), "falsy decompose #{i}");
    }
}
