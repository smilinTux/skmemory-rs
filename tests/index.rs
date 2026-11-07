//! SKMEM-R-03: the SQLite index against the Python backend.
//!
//! `tests/golden/index/py_index.json` is produced by `tests/golden/index_parity.py`
//! with the real Python `SQLiteBackend` over the golden corpus (hermetic here:
//! no Python needed). The same corpus is indexed with the Rust engine and every
//! `memories` column, the FTS sidecar, the schema text, bm25 result order and
//! the usage counters are diffed. With `SKMEM_PYTHON` set (an interpreter that
//! can import skmemory) the golden file is also regenerated and compared, and
//! Rust and Python open one `index.db` in turn.

use serde_json::{json, Value};
use skmem::flat::Store;
use skmem::index::{
    initial_importance, sanitize_match, Index, MemoryRow, SearchFilters, REINFORCE_IMPORTANCE_STEP,
};
use skmem::model::Memory;
use std::fs;
use std::path::{Path, PathBuf};

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var_os("SKMEM_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("skmem-idx-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn expected() -> Value {
    serde_json::from_str(&fs::read_to_string(golden().join("index/py_index.json")).unwrap())
        .unwrap()
}

/// Copy the golden corpus into layer directories, as `index_parity.py` does.
fn build_tree(root: &Path) -> Store {
    let store = Store::open(root).unwrap();
    let mut files: Vec<PathBuf> = fs::read_dir(golden().join("out"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    for f in files {
        let raw = fs::read(&f).unwrap();
        if let Ok(m) = Memory::from_stored_json(&raw) {
            fs::write(store.path_for(&m).unwrap(), raw).unwrap();
        }
    }
    store
}

/// Index every file in the order Python's script uses (layer, sorted name).
fn index_sorted(idx: &Index, store: &Store) {
    for l in skmem::model::Layer::ALL {
        let mut fs_: Vec<PathBuf> = fs::read_dir(store.layer_dir(l))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        fs_.sort();
        for p in fs_ {
            let m = skmem::model::Memory::from_stored_json(&fs::read(&p).unwrap()).unwrap();
            idx.index_memory(&m, &p).unwrap();
        }
    }
}

fn row_json(r: &MemoryRow, root: &Path) -> Value {
    let rel = Path::new(&r.file_path).strip_prefix(root).unwrap();
    json!({
        "id": r.id, "title": r.title, "layer": r.layer, "role": r.role, "tags": r.tags,
        "source": r.source, "source_ref": r.source_ref, "summary": r.summary,
        "content_preview": r.content_preview,
        "emotional_intensity": r.emotional_intensity, "emotional_valence": r.emotional_valence,
        "emotional_labels": r.emotional_labels, "cloud9_achieved": r.cloud9_achieved,
        "importance": r.importance, "access_count": r.access_count,
        "last_accessed": r.last_accessed, "parent_id": r.parent_id,
        "related_ids": r.related_ids, "created_at": r.created_at, "updated_at": r.updated_at,
        "file_path": rel.to_string_lossy(), "content_hash": r.content_hash, "kind": r.kind,
        "outcome": r.outcome, "event_at": r.event_at, "valid_from": r.valid_from,
        "valid_to": r.valid_to, "supersedes_id": r.supersedes_id,
        "source_tier": r.source_tier, "schema_version": r.schema_version,
    })
}

/// Every column of every row, compared field by field (floats numerically).
fn diff_rows(got: &[Value], want: &Value, what: &str) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (g, w) in got.iter().zip(want) {
        let wo = w.as_object().unwrap();
        let go = g.as_object().unwrap();
        assert_eq!(go.len(), wo.len(), "{what}: column count");
        for (k, wv) in wo {
            let gv = &go[k];
            let same = match (gv.as_f64(), wv.as_f64()) {
                (Some(a), Some(b)) => a == b,
                _ => gv == wv,
            };
            assert!(
                same,
                "{what}: id={} column {k}: rust={gv} python={wv}",
                wo["id"]
            );
        }
    }
}

fn all_json(idx: &Index, root: &Path) -> Vec<Value> {
    idx.all_rows(&owner())
        .unwrap()
        .iter()
        .map(|r| row_json(r, root))
        .collect()
}

fn schema_json(idx: &Index) -> Vec<Value> {
    let mut st = idx
        .connection(&own())
        .prepare(
            "SELECT type, name, tbl_name, sql FROM sqlite_master \
             WHERE name NOT LIKE 'sqlite_%' AND name NOT LIKE 'memory_fts_%' ORDER BY type, name",
        )
        .unwrap();
    st.query_map([], |r| {
        Ok(
            json!({"type": r.get::<_, String>(0)?, "name": r.get::<_, String>(1)?,
                  "tbl_name": r.get::<_, String>(2)?, "sql": r.get::<_, Option<String>>(3)?}),
        )
    })
    .unwrap()
    .map(|r| r.unwrap())
    .collect()
}

fn fts_meta_json(idx: &Index) -> Vec<Value> {
    let mut st = idx
        .connection(&own())
        .prepare("SELECT rowid, id, content_hash, context_tag FROM fts_meta ORDER BY rowid")
        .unwrap();
    st.query_map([], |r| {
        Ok(
            json!({"rowid": r.get::<_, i64>(0)?, "id": r.get::<_, String>(1)?,
                  "content_hash": r.get::<_, String>(2)?, "context_tag": r.get::<_, String>(3)?}),
        )
    })
    .unwrap()
    .map(|r| r.unwrap())
    .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn schema_rows_fts_and_search_match_python() {
    let want = expected();
    let dir = scratch("parity");
    let store = build_tree(&dir.join("tree"));
    let idx = Index::open(store.root(), true).unwrap();
    assert!(idx.fts_ok());
    index_sorted(&idx, &store);

    assert_eq!(
        Value::Array(schema_json(&idx)),
        want["schema"],
        "schema text"
    );
    diff_rows(&all_json(&idx, store.root()), &want["rows"], "memories");
    assert_eq!(
        Value::Array(fts_meta_json(&idx)),
        want["fts_meta"],
        "fts_meta"
    );
    let cov = idx.fts_status(&own()).unwrap().unwrap();
    assert_eq!(
        (cov.total, cov.indexed, cov.missing),
        (75, 75, 0),
        "coverage"
    );

    let mut non_empty = 0;
    for q in want["queries"].as_array().unwrap() {
        let tags: Vec<String> = q.get("tags").map(strs).unwrap_or_default();
        let ctx: Option<Vec<String>> = q.get("context_tag").map(strs);
        let f = SearchFilters {
            layer: q.get("layer").and_then(Value::as_str),
            tags: &tags,
            source: q.get("source").and_then(Value::as_str),
            context_tags: ctx.as_deref(),
            as_of: q.get("as_of").and_then(Value::as_str),
        };
        let got: Vec<String> = idx
            .search_lexical(q["q"].as_str().unwrap(), 10, &f, &owner())
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(got, strs(&q["ids"]), "bm25 order for {q}");
        non_empty += usize::from(!got.is_empty());
    }
    assert!(non_empty > 20, "query set must exercise real hits");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn counters_reindex_and_backfill_match_python() {
    let want = expected();
    let dir = scratch("counters");
    let store = build_tree(&dir.join("tree"));
    let idx = Index::open(store.root(), true).unwrap();
    index_sorted(&idx, &store);

    let ids = strs(&want["ids"]);
    let id3: Vec<&str> = ids[..3].iter().map(String::as_str).collect();
    let id2: Vec<&str> = ids[..2].iter().map(String::as_str).collect();
    let id35: Vec<&str> = ids[3..5].iter().map(String::as_str).collect();
    let fut = "2999-01-01T00:00:00+00:00";
    let bumps = [
        idx.bump_access(&id3, Some(fut), None, &owner()),
        idx.bump_access(&id3, Some(fut), None, &owner()),
        idx.bump_access(&id2, None, Some(3600.0), &owner()),
        idx.bump_access(
            &id35,
            Some("2026-01-01T00:00:00+00:00"),
            Some(3600.0),
            &owner(),
        ),
    ];
    assert_eq!(json!(bumps), want["bumps"], "bump row counts");

    let legacy = want["legacy_id"].as_str().unwrap();
    idx.connection(&own())
        .execute(
            "UPDATE memories SET importance = 0.5 WHERE id = ?",
            [legacy],
        )
        .unwrap();
    let six: Vec<&str> = ids[..6].iter().map(String::as_str).collect();
    let usage = idx.get_usage(&six, &owner());
    for (k, v) in want["usage"].as_object().unwrap() {
        let (c, la) = &usage[k];
        assert_eq!(*c, v["access_count"].as_i64().unwrap(), "usage count {k}");
        assert_eq!(la.as_deref(), v["last_accessed"].as_str(), "usage ts {k}");
    }
    diff_rows(
        &all_json(&idx, store.root()),
        &want["after_bump"],
        "after bump",
    );

    let rep = idx.reindex(&store, true, &own()).unwrap();
    assert_eq!(json!(rep.indexed), want["reindex_count"]);
    diff_rows(
        &all_json(&idx, store.root()),
        &want["after_reindex"],
        "after reindex",
    );

    let l2 = want["legacy2_id"].as_str().unwrap();
    idx.connection(&own())
        .execute("UPDATE memories SET importance = 0.5 WHERE id = ?", [l2])
        .unwrap();
    let b = idx.backfill_importance(500, &own()).unwrap();
    let wb = &want["backfill_importance"];
    assert_eq!(
        (b.total, b.scanned, b.updated, b.unchanged),
        (
            wb["total"].as_i64().unwrap(),
            wb["scanned"].as_i64().unwrap(),
            wb["updated"].as_i64().unwrap(),
            wb["unchanged"].as_i64().unwrap()
        )
    );
    diff_rows(
        &all_json(&idx, store.root()),
        &want["after_backfill"],
        "after backfill",
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn upsert_preserves_usage_and_reinforcement_step() {
    let dir = scratch("upsert");
    let store = Store::open(dir.join("m")).unwrap();
    let idx = Index::open(store.root(), true).unwrap();
    let mut m = Memory::new("alpha", "first body");
    m.emotional.intensity = 5.0;
    store.save(&m).unwrap();
    let p = store.path_for(&m).unwrap();
    idx.index_memory(&m, &p).unwrap();
    let start = idx.get(&m.id, &owner()).unwrap().unwrap();
    assert_eq!(start.importance, initial_importance("short-term", 5.0));
    assert_eq!(start.access_count, 0);

    assert_eq!(
        idx.bump_access(&[&m.id], Some("2030-01-01T00:00:00+00:00"), None, &owner()),
        1
    );
    let bumped = idx.get(&m.id, &owner()).unwrap().unwrap();
    let step = start.importance + REINFORCE_IMPORTANCE_STEP * (1.0 - start.importance);
    assert!((bumped.importance - step).abs() < 1e-12);
    assert_eq!(bumped.access_count, 1);

    // Re-save with new content: content columns change, usage state does not.
    m.content = "second body".into();
    m.title = "beta".into();
    idx.index_memory(&m, &p).unwrap();
    let after = idx.get(&m.id, &owner()).unwrap().unwrap();
    assert_eq!(after.title, "beta");
    assert_eq!(after.access_count, 1);
    assert_eq!(after.last_accessed, bumped.last_accessed);
    assert_eq!(after.importance, bumped.importance);
    assert_eq!(
        idx.search_lexical("second", 5, &SearchFilters::default(), &owner())
            .unwrap()
            .len(),
        1
    );
    assert!(idx
        .search_lexical("first", 5, &SearchFilters::default(), &owner())
        .unwrap()
        .is_empty());

    // Debounce: last_accessed is in the future, so an hour guard skips it.
    assert_eq!(idx.bump_access(&[&m.id], None, Some(3600.0), &owner()), 0);
    // Empty and duplicate ids are ignored and never fail.
    assert_eq!(
        idx.bump_access(&["", &m.id, &m.id], None, None, &owner()),
        1
    );
    assert_eq!(idx.bump_access(&[], None, None, &owner()), 0);
    assert_eq!(idx.bump_access(&["nope"], None, None, &owner()), 0);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn importance_matches_python_rounding() {
    assert_eq!(initial_importance("short-term", 0.0), 0.3);
    assert_eq!(initial_importance("long-term", 10.0), 1.0);
    assert_eq!(initial_importance("mid-term", 8.5), 0.755);
    assert_eq!(initial_importance("mid-term", 99.0), 0.8);
    assert_eq!(initial_importance("weird", 0.0), 0.5);
    assert_eq!(initial_importance("mid-term", f64::NAN), 0.5);
}

#[test]
fn sanitize_match_is_inert() {
    assert_eq!(
        sanitize_match("prb-6f069c5e", " OR ").unwrap(),
        "\"prb 6f069c5e\""
    );
    assert_eq!(
        sanitize_match("a NEAR(b) \"c\"", " AND ").unwrap(),
        "\"a\" AND \"NEAR b\" AND \"c\""
    );
    assert!(sanitize_match("  ()  -- ", " OR ").is_none());
    let many = (0..50)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(
        sanitize_match(&many, " OR ")
            .unwrap()
            .matches(" OR ")
            .count(),
        31
    );
}

#[test]
fn vaulted_store_indexes_no_plaintext_and_refuses_fts() {
    let dir = scratch("vault");
    let store = Store::open(dir.join("m")).unwrap();
    // First opened unencrypted: FTS tables hold plaintext.
    {
        let idx = Index::open(store.root(), true).unwrap();
        let m = Memory::new("t", &format!("{} SECRETSENTINELXYZ body", "p".repeat(300)));
        idx.index_memory(&m, &store.path_for(&m).unwrap()).unwrap();
        assert_eq!(
            idx.search_lexical("SECRETSENTINELXYZ", 5, &SearchFilters::default(), &owner())
                .unwrap()
                .len(),
            1
        );
    }
    let idx = Index::open(store.root(), false).unwrap();
    assert!(!idx.fts_ok() && !idx.fts_enabled());
    let n: i64 = idx
        .connection(&own())
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'memory_fts%' OR name='fts_meta' OR name='memories_fts_ad'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0, "FTS artefacts dropped");
    assert!(idx
        .search_lexical("SECRETSENTINELXYZ", 5, &SearchFilters::default(), &owner())
        .unwrap()
        .is_empty());
    assert!(idx.fts_status(&own()).unwrap().is_none());
    // New writes add no FTS text either, and the db file carries no plaintext
    // beyond the 150-char preview the plain index already keeps.
    let m = Memory::new("v", &format!("{} VAULTSENTINEL2", "x".repeat(400)));
    idx.index_memory(&m, &store.path_for(&m).unwrap()).unwrap();
    idx.connection(&own())
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    drop(idx);
    let bytes = fs::read(store.root().join("index.db")).unwrap();
    let has = |needle: &str| bytes.windows(needle.len()).any(|w| w == needle.as_bytes());
    assert!(!has("SECRETSENTINELXYZ"), "scrubbed by VACUUM");
    assert!(!has("VAULTSENTINEL2"), "tail of content never indexed");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn open_is_idempotent_migrates_old_schema_and_survives_concurrent_openers() {
    let dir = scratch("open");
    let db = dir.join("index.db");
    // An old Python index: only the original columns.
    {
        let c = rusqlite::Connection::open(&db).unwrap();
        c.execute_batch(
            "CREATE TABLE memories (id TEXT PRIMARY KEY, title TEXT NOT NULL, layer TEXT NOT NULL, \
             role TEXT NOT NULL DEFAULT 'general', tags TEXT NOT NULL DEFAULT '', \
             source TEXT NOT NULL DEFAULT 'manual', source_ref TEXT NOT NULL DEFAULT '', \
             summary TEXT NOT NULL DEFAULT '', content_preview TEXT NOT NULL DEFAULT '', \
             emotional_intensity REAL NOT NULL DEFAULT 0.0, emotional_valence REAL NOT NULL DEFAULT 0.0, \
             emotional_labels TEXT NOT NULL DEFAULT '', cloud9_achieved INTEGER NOT NULL DEFAULT 0, \
             parent_id TEXT, related_ids TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, \
             updated_at TEXT NOT NULL, file_path TEXT NOT NULL); \
             INSERT INTO memories(id,title,layer,created_at,updated_at,file_path) \
             VALUES('old1','t','long-term','2020-01-01T00:00:00+00:00','2020-01-01T00:00:00+00:00','/x/old1.json'); \
             CREATE VIEW active_memories AS SELECT id FROM memories;",
        )
        .unwrap();
    }
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let db = db.clone();
            std::thread::spawn(move || {
                for _ in 0..5 {
                    Index::open_db(&db, true).expect("concurrent open");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let idx = Index::open_db(&db, true).unwrap();
    let r = idx.get("old1", &owner()).unwrap().unwrap();
    assert_eq!(
        (r.importance, r.access_count, r.kind.as_str()),
        (0.5, 0, "episode")
    );
    assert_eq!(r.source_tier, "self-generated");
    assert_eq!(r.schema_version, 1);
    // A node's own view is left alone (Python recreates it; Rust never drops).
    let view: String = idx
        .connection(&own())
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name='active_memories'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(view.contains("SELECT id FROM memories"));
    // Same schema again changes nothing.
    let before = schema_json(&idx);
    drop(idx);
    let again = Index::open_db(&db, true).unwrap();
    assert_eq!(schema_json(&again), before);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn reindex_keeps_orphans_unless_forced_and_restores_usage_atomically() {
    let dir = scratch("orphans");
    let store = Store::open(dir.join("m")).unwrap();
    let idx = Index::open(store.root(), true).unwrap();
    let a = Memory::new("a", "alpha one");
    let b = Memory::new("b", "beta two");
    store.save(&a).unwrap();
    store.save(&b).unwrap();
    idx.index_memory(&a, &store.path_for(&a).unwrap()).unwrap();
    idx.index_memory(&b, &store.path_for(&b).unwrap()).unwrap();
    // An index-only row (no flat file) and a tombstoned one.
    let orphan = Memory::new("orph", "orphan text");
    idx.index_memory(&orphan, &store.path_for(&orphan).unwrap())
        .unwrap();
    let dead = Memory::new("dead", "forgotten text");
    idx.index_memory(&dead, &store.path_for(&dead).unwrap())
        .unwrap();
    store
        .tombstone(&dead.id, "2026-01-01T00:00:00+0000", "test", None, &own())
        .unwrap();
    idx.bump_access(
        &[&a.id, &a.id],
        Some("2031-01-01T00:00:00+00:00"),
        None,
        &owner(),
    );
    idx.bump_access(
        &[&orphan.id],
        Some("2031-01-01T00:00:00+00:00"),
        None,
        &owner(),
    );
    let imp_a = idx.get(&a.id, &owner()).unwrap().unwrap().importance;

    let rep = idx.reindex(&store, false, &own()).unwrap();
    assert_eq!((rep.indexed, rep.orphans_kept), (2, 1));
    assert!(
        idx.get(&orphan.id, &owner()).unwrap().is_some(),
        "orphan kept"
    );
    assert!(
        idx.get(&dead.id, &owner()).unwrap().is_none(),
        "tombstoned row dropped"
    );
    let ra = idx.get(&a.id, &owner()).unwrap().unwrap();
    assert_eq!(ra.access_count, 1);
    assert_eq!(ra.importance, imp_a);
    assert_eq!(
        ra.last_accessed.as_deref(),
        Some("2031-01-01T00:00:00+00:00")
    );
    assert_eq!(
        idx.get(&orphan.id, &owner()).unwrap().unwrap().access_count,
        1,
        "orphan counters untouched"
    );
    // FTS follows the rebuild: the dropped row is gone, the kept row remains.
    let f = SearchFilters::default();
    assert!(idx
        .search_lexical("forgotten", 5, &f, &owner())
        .unwrap()
        .is_empty());
    assert_eq!(
        idx.search_lexical("alpha", 5, &f, &owner()).unwrap().len(),
        1
    );
    assert_eq!(
        idx.search_lexical("orphan", 5, &f, &owner()).unwrap().len(),
        1
    );

    idx.reindex(&store, true, &own()).unwrap();
    assert!(
        idx.get(&orphan.id, &owner()).unwrap().is_none(),
        "force drops orphans"
    );
    assert_eq!(idx.count(&owner()).unwrap(), 2);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn fts_backfill_indexes_rows_missing_from_a_python_index() {
    let dir = scratch("backfill");
    let store = Store::open(dir.join("m")).unwrap();
    let m = Memory::new("lazy", "needle in haystack");
    store.save(&m).unwrap();
    let idx = Index::open(store.root(), true).unwrap();
    idx.index_memory(&m, &store.path_for(&m).unwrap()).unwrap();
    // Simulate a Python index written before the FTS arm existed.
    idx.connection(&own())
        .execute_batch("DELETE FROM memory_fts; DELETE FROM fts_meta")
        .unwrap();
    assert_eq!(idx.fts_status(&own()).unwrap().unwrap().missing, 1);
    assert!(idx
        .search_lexical("needle", 5, &SearchFilters::default(), &owner())
        .unwrap()
        .is_empty());
    let r = idx.fts_backfill(&store, None, None, &own()).unwrap();
    assert!(r.complete && r.indexed == 1);
    assert_eq!(
        idx.search_lexical("needle", 5, &SearchFilters::default(), &owner())
            .unwrap()
            .len(),
        1
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_is_bounded() {
    let dir = scratch("wal");
    let idx = Index::open(dir.as_path(), true).unwrap();
    let c = idx.connection(&own());
    let lim: i64 = c
        .query_row("PRAGMA journal_size_limit", [], |r| r.get(0))
        .unwrap();
    let auto: i64 = c
        .query_row("PRAGMA wal_autocheckpoint", [], |r| r.get(0))
        .unwrap();
    let mode: String = c
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!((lim, auto, mode.as_str()), (64 * 1024 * 1024, 1000, "wal"));
    assert!(idx.wal_status().wal_ok);
    let _ = fs::remove_dir_all(dir);
}

/// Regenerates the golden file with Python and checks it is unchanged, then
/// has Rust and Python open one index.db in turn. Needs SKMEM_PYTHON.
#[test]
fn python_cross_open_and_golden_is_current() {
    let Ok(py) = std::env::var("SKMEM_PYTHON") else {
        eprintln!("SKMEM_PYTHON unset: skipping Python cross-open");
        return;
    };
    let dir = scratch("xopen");
    let first = strs(&expected()["ids"])[0].clone();
    let regen = dir.join("regen.json");
    let st = std::process::Command::new(&py)
        .arg(golden().join("index_parity.py"))
        .arg(golden().join("out"))
        .arg(dir.join("work"))
        .arg(&regen)
        .status()
        .expect("run python");
    assert!(st.success());
    let a: Value = serde_json::from_str(&fs::read_to_string(&regen).unwrap()).unwrap();
    assert_eq!(
        a,
        expected(),
        "py_index.json is stale: rerun index_parity.py"
    );

    // Rust indexes, then Python opens the SAME file, reindexes and reads it.
    let store = build_tree(&dir.join("tree"));
    {
        let idx = Index::open(store.root(), true).unwrap();
        index_sorted(&idx, &store);
        idx.bump_access(&[&first], Some("2030-01-01T00:00:00+00:00"), None, &owner());
    }
    let script = "import sys\nfrom skmemory.backends.sqlite_backend import SQLiteBackend\n\
        b=SQLiteBackend(base_path=sys.argv[1])\nu=b.get_usage([sys.argv[2]])\n\
        n=b.reindex(force=True)\nu2=b.get_usage([sys.argv[2]])\n\
        assert u==u2, (u,u2)\nassert b.fts_status(&own())['indexed']==n\nprint(n)";
    let o = std::process::Command::new(&py)
        .args(["-c", script])
        .arg(store.root())
        .arg(&first)
        .output()
        .expect("run python");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    // And Rust still opens what Python rewrote, without changing the schema.
    let idx = Index::open(store.root(), true).unwrap();
    assert!(idx.count(&owner()).unwrap() >= 70);
    assert_eq!(
        idx.get_usage(&[first.as_str()], &owner())
            .get(&first)
            .map(|u| u.0),
        Some(1)
    );
    let _ = fs::remove_dir_all(dir);
}

/// Reindex timing on a generated tree. `cargo test --release -- --ignored
/// --nocapture bench_reindex` with SKMEM_BENCH_N (default 10000) and
/// SKMEM_BENCH_DIR (a scratch directory).
#[test]
#[ignore]
fn bench_reindex() {
    let n: usize = std::env::var("SKMEM_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let dir = PathBuf::from(std::env::var("SKMEM_BENCH_DIR").expect("SKMEM_BENCH_DIR"));
    let _ = fs::remove_dir_all(&dir);
    let store = Store::open(dir.join("m")).unwrap();
    let words: Vec<String> = (0..4000)
        .map(|i| format!("w{i:04}x{}", (i * 7919) % 1013))
        .collect();
    let t = std::time::Instant::now();
    for i in 0..n {
        let mut body = String::new();
        let mut s = i as u64 * 2654435761 + 1;
        for _ in 0..120 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            body.push_str(&words[(s >> 33) as usize % words.len()]);
            body.push(' ');
        }
        let mut m = Memory::new(&format!("title {i} {}", words[i % words.len()]), &body);
        m.layer = skmem::model::Layer::ALL[i % 3];
        m.emotional.intensity = (i % 11) as f64;
        m.tags = vec![format!("t{}", i % 50), "bench".into()];
        store.save(&m).unwrap();
    }
    eprintln!("generated {n} files in {:?}", t.elapsed());
    let idx = Index::open(store.root(), true).unwrap();
    let t = std::time::Instant::now();
    let r = idx.reindex(&store, false, &own()).unwrap();
    let cold = t.elapsed();
    eprintln!("reindex cold: {} rows in {cold:?}", r.indexed);
    let t = std::time::Instant::now();
    let r = idx.reindex(&store, false, &own()).unwrap();
    eprintln!(
        "reindex rebuild over existing index: {} rows in {:?}",
        r.indexed,
        t.elapsed()
    );
    let t = std::time::Instant::now();
    let hits = idx
        .search_lexical("w0042x1013 bench", 10, &SearchFilters::default(), &owner())
        .unwrap();
    eprintln!("search: {} hits in {:?}", hits.len(), t.elapsed());
    let ids: Vec<String> = idx
        .all_rows(&owner())
        .unwrap()
        .into_iter()
        .take(500)
        .map(|r| r.id)
        .collect();
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let t = std::time::Instant::now();
    let u = idx.bump_access(&refs, None, None, &owner());
    eprintln!("bump 500: {u} rows in {:?}", t.elapsed());
}

#[test]
fn word_chars_matches_python_w() {
    let raw = fs::read_to_string(golden().join("word_chars.json")).unwrap();
    let cases: Value = serde_json::from_str(&raw).unwrap();
    for case in cases.as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let want: Vec<String> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        assert_eq!(skmem::index::word_chars(input), want, "input {input:?}");
    }
}

/// The owner scope: no reader declared and no `SKMEMORY_READER` ceiling.
fn own() -> skmem::audience::OwnerScope {
    skmem::audience::OwnerScope::resolve(&skmem::env::Env::empty()).unwrap()
}

fn owner() -> skmem::audience::ReaderScope {
    own().reader_scope().clone()
}
