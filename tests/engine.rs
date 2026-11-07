//! SKMEM-R-07C: `skmem::engine::Engine` composition, deadline, per-arm status
//! and the read-only guarantees. Hermetic: fake arms, a scratch store, no
//! network and no Postgres (the Postgres arms are in `tests/engine_pg.rs`).

use skmem::audience::{AudienceConfig, OwnerScope, ReaderScope};
use skmem::engine::{
    Arms, Attach, Engine, EngineOptions, Filters, GraphSource, MemoryArm, SearchPath, SearchRequest,
};
use skmem::env::Env;
use skmem::fusion::ArmStatus;
use skmem::index::Index;
use skmem::model::Memory;
use skmem::pytime::Dt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn tmp(tag: &str) -> PathBuf {
    let base = std::env::var_os("SKMEM_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("skmem-r07c-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn own() -> OwnerScope {
    OwnerScope::resolve(&Env::empty()).unwrap()
}

fn public() -> ReaderScope {
    ReaderScope::resolve(Some("@public"), &Env::empty(), &AudienceConfig::empty())
}

fn mem(id: &str, ctx: &str, title: &str, created: &str) -> Memory {
    let mut m = Memory::new(title, &format!("{title} body text for {id}"));
    m.id = id.into();
    m.context_tag = ctx.into();
    m.created_at = created.into();
    m.updated_at = created.into();
    m
}

fn seeded(tag: &str, arms: Arms) -> (Engine, PathBuf) {
    let dir = tmp(tag);
    let root = dir.join("memory");
    let e = Engine::open(
        &root,
        EngineOptions {
            read_only: Some(false),
            ..EngineOptions::new(arms)
        },
    )
    .unwrap();
    for m in [
        mem(
            "a1",
            "@public",
            "amberfox river delta",
            "2026-10-01T10:00:00+00:00",
        ),
        mem(
            "a2",
            "@public",
            "amberfox lake notes",
            "2026-10-02T10:00:00+00:00",
        ),
        mem(
            "p1",
            "@chef-only",
            "amberfox private river",
            "2026-10-03T10:00:00+00:00",
        ),
    ] {
        e.save(&m, &own()).unwrap();
    }
    (e, root)
}

fn req(q: &str) -> SearchRequest<'_> {
    SearchRequest {
        query: q,
        limit: 10,
        ..SearchRequest::default()
    }
}

fn status(out: &skmem::engine::SearchOutput, arm: &str) -> ArmStatus {
    out.report
        .arms
        .iter()
        .find(|a| a.name == arm)
        .unwrap()
        .status
}

fn ids(out: &skmem::engine::SearchOutput) -> Vec<&str> {
    out.memories.iter().map(|m| m.id.as_str()).collect()
}

/// A fake arm: returns fixed memories after an optional delay.
struct Fake {
    ret: Vec<Memory>,
    delay: Duration,
    fail: Option<String>,
    calls: Arc<AtomicUsize>,
}

impl Fake {
    fn new(ret: Vec<Memory>) -> Fake {
        Fake {
            ret,
            delay: Duration::ZERO,
            fail: None,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl MemoryArm for Fake {
    fn search(
        &mut self,
        _q: &str,
        _l: usize,
        _f: &Filters,
        _s: &ReaderScope,
    ) -> Result<Vec<Memory>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        match &self.fail {
            Some(e) => Err(e.clone()),
            None => Ok(self.ret.clone()),
        }
    }
}

fn env_deadline(s: &str) -> Env {
    Env::empty().with("SKMEMORY_SEARCH_DEADLINE_S", s)
}

#[test]
fn sqlite_only_engine_reports_every_arm_honestly() {
    let (e, _) = seeded("status", Arms::none());
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &Env::empty())
        .unwrap();
    assert_eq!(out.report.path, SearchPath::Fused);
    assert_eq!(status(&out, "semantic"), ArmStatus::Unavailable);
    // No BM25 backend: `lexical` is the SQLite LIKE search, as in Python.
    assert_eq!(status(&out, "lexical"), ArmStatus::Ok);
    assert_eq!(status(&out, "lexical_local"), ArmStatus::Ok);
    assert_eq!(status(&out, "graph"), ArmStatus::Unavailable);
    assert_eq!(status(&out, "temporal"), ArmStatus::Abstained);
    assert_eq!(out.memories.len(), 3);
    assert!(out.report.degraded.is_none());
}

#[test]
fn a_slow_arm_times_out_and_the_others_still_answer() {
    let mut slow = Fake::new(vec![mem(
        "a1",
        "@public",
        "amberfox river delta",
        "2026-10-01T10:00:00+00:00",
    )]);
    slow.delay = Duration::from_secs(3);
    let arms = Arms {
        semantic: Attach::ready(Box::new(slow)),
        lexical: Attach::Absent,
        graph: Attach::Absent,
    };
    let (e, _) = seeded("timeout", arms);
    let t = Instant::now();
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &env_deadline("0.3"))
        .unwrap();
    assert!(
        t.elapsed() < Duration::from_millis(2500),
        "waited {:?}",
        t.elapsed()
    );
    assert_eq!(status(&out, "semantic"), ArmStatus::Timeout);
    assert_eq!(out.report.degraded, Some("arms timed out"));
    assert_eq!(out.memories.len(), 3, "the local arms still fused a page");
    assert_eq!(out.report.path, SearchPath::Fused);

    // The abandoned call still holds the arm: the next search does not queue
    // behind it, it reports timeout at once.
    let t = Instant::now();
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &env_deadline("5"))
        .unwrap();
    assert!(
        t.elapsed() < Duration::from_millis(1500),
        "queued: {:?}",
        t.elapsed()
    );
    assert_eq!(status(&out, "semantic"), ArmStatus::Timeout);
    assert!(out.report.arms[0].detail.contains("still running"));
}

#[test]
fn a_failed_arm_is_failed_never_empty_and_does_not_stop_the_search() {
    let mut bad = Fake::new(vec![]);
    bad.fail = Some("boom".into());
    let arms = Arms {
        semantic: Attach::ready(Box::new(bad)),
        lexical: Attach::Failed("pg down".into()),
        graph: Attach::Disabled(
            "disabled by config (SKMEMORY_GRAPH_ARM is off, the default)".into(),
        ),
    };
    let (e, _) = seeded("failed", arms);
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &Env::empty())
        .unwrap();
    assert_eq!(status(&out, "semantic"), ArmStatus::Failed);
    assert_eq!(out.report.arms[0].detail, "boom");
    assert_eq!(status(&out, "lexical"), ArmStatus::Failed);
    assert_eq!(status(&out, "graph"), ArmStatus::Disabled);
    assert_eq!(out.memories.len(), 3);
    assert!(out.report.degraded.is_none(), "failed is not a timeout");
}

#[test]
fn nothing_fused_after_a_timeout_never_takes_the_legacy_path() {
    let mut slow = Fake::new(vec![]);
    slow.delay = Duration::from_secs(2);
    let arms = Arms {
        semantic: Attach::ready(Box::new(slow)),
        lexical: Attach::Absent,
        graph: Attach::Absent,
    };
    let (e, _) = seeded("emptytimeout", arms);
    let out = e
        .search(
            &req("zzzzunmatched"),
            own().reader_scope(),
            &env_deadline("0.2"),
        )
        .unwrap();
    assert!(out.memories.is_empty());
    assert_eq!(out.report.path, SearchPath::Empty);
    assert_eq!(out.report.degraded, Some("arms timed out"));
}

#[test]
fn an_arm_that_returns_a_hidden_row_cannot_leak_it() {
    // A misbehaving backend hands back a private memory: the reader scope drops
    // it before fusion, whichever arm it came from.
    let private = mem(
        "p1",
        "@chef-only",
        "amberfox private river",
        "2026-10-03T10:00:00+00:00",
    );
    let arms = Arms {
        semantic: Attach::ready(Box::new(Fake::new(vec![private.clone()]))),
        lexical: Attach::ready(Box::new(Fake::new(vec![private]))),
        graph: Attach::Absent,
    };
    let (e, _) = seeded("leak", arms);
    let out = e
        .search(&req("amberfox"), &public(), &Env::empty())
        .unwrap();
    assert!(!ids(&out).contains(&"p1"), "{:?}", ids(&out));
    assert!(!ids(&out).is_empty());
    for a in &out.report.arms {
        assert!(!a
            .ids
            .iter()
            .any(|i| i == "p1" && a.name != "semantic" && a.name != "lexical"));
    }
}

struct FakeGraph(Vec<String>);

impl GraphSource for FakeGraph {
    fn search(
        &mut self,
        _: &str,
        _: usize,
        _: &Filters,
        _: &ReaderScope,
    ) -> Result<Vec<String>, String> {
        Ok(self.0.clone())
    }
}

#[test]
fn graph_ids_are_hydrated_from_the_flat_store_and_rechecked() {
    let arms = Arms {
        semantic: Attach::Absent,
        lexical: Attach::Absent,
        graph: Attach::ready(Box::new(FakeGraph(vec![
            "p1".into(),
            "nope".into(),
            "a2".into(),
        ]))),
    };
    let (e, _) = seeded("graph", arms);
    // The owner sees p1; a stale graph id ("nope") is skipped, not emitted.
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &Env::empty())
        .unwrap();
    let g = out.report.arms.iter().find(|a| a.name == "graph").unwrap();
    assert_eq!(g.status, ArmStatus::Ok);
    assert_eq!(g.ids, ["p1", "a2"]);
    // A public reader loses p1 in the graph arm too.
    let out = e
        .search(&req("amberfox"), &public(), &Env::empty())
        .unwrap();
    let g = out.report.arms.iter().find(|a| a.name == "graph").unwrap();
    assert_eq!(g.ids, ["a2"]);
}

#[test]
fn temporal_arm_fires_only_on_a_time_window() {
    let (e, _) = seeded("temporal", Arms::none());
    let mut r = req("what happened yesterday");
    r.now = Some(Dt::parse("2026-10-03T12:00:00+00:00").unwrap());
    let out = e.search(&r, own().reader_scope(), &Env::empty()).unwrap();
    let t = out
        .report
        .arms
        .iter()
        .find(|a| a.name == "temporal")
        .unwrap();
    assert_eq!(t.status, ArmStatus::Ok);
    assert_eq!(t.ids, ["a2"], "yesterday relative to 2026-10-03 is the 2nd");
    let out = e
        .search(&req("amberfox"), own().reader_scope(), &Env::empty())
        .unwrap();
    assert_eq!(status(&out, "temporal"), ArmStatus::Abstained);
}

// ------------------------------------------------------------ read-only (a)

fn sha(path: &Path) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(std::fs::read(path).unwrap()).to_vec()
}

fn files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = walk(dir);
    v.sort();
    v
}

fn walk(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p.to_string_lossy().into_owned());
            }
        }
    }
    out
}

#[test]
fn a_read_only_engine_never_writes_index_db() {
    let (owner_engine, root) = seeded("ro", Arms::none());
    drop(owner_engine); // close: the WAL is checkpointed
    let db = root.join("index.db");
    let before = (
        sha(&db),
        std::fs::metadata(&db).unwrap().modified().unwrap(),
    );
    let ro = Engine::open(
        &root,
        EngineOptions {
            read_only: Some(true),
            ..EngineOptions::new(Arms::none())
        },
    )
    .unwrap();
    assert!(ro.is_read_only());
    assert!(ro.index().is_read_only());
    for q in ["amberfox", "river", "what happened yesterday", "zzzz"] {
        let out = ro
            .search(&req(q), own().reader_scope(), &Env::empty())
            .unwrap();
        assert!(out.memories.len() <= 3);
    }
    let out = ro
        .search(&req("amberfox"), &public(), &Env::empty())
        .unwrap();
    assert_eq!(ids(&out).len(), 2, "public sees a1 and a2 only");
    // A bump is refused on the handle itself.
    assert_eq!(
        ro.index()
            .bump_access(&["a1"], None, None, own().reader_scope()),
        0
    );
    let after = (
        sha(&db),
        std::fs::metadata(&db).unwrap().modified().unwrap(),
    );
    assert_eq!(before, after, "index.db changed under a read-only engine");
    // Writes are refused.
    let e = ro.save(
        &mem("z", "@public", "t", "2026-10-04T00:00:00+00:00"),
        &own(),
    );
    assert!(matches!(e, Err(skmem::engine::EngineError::ReadOnly)));
}

#[test]
fn a_read_only_engine_creates_nothing_in_a_missing_store() {
    let dir = tmp("ro-missing");
    let root = dir.join("memory");
    let ro = Engine::open(
        &root,
        EngineOptions {
            read_only: Some(true),
            ..EngineOptions::new(Arms::none())
        },
    )
    .unwrap();
    let out = ro
        .search(&req("anything"), &public(), &Env::empty())
        .unwrap();
    assert!(out.memories.is_empty());
    assert!(!root.exists(), "a reader created {:?}", files(&dir));
}

#[test]
fn open_readonly_index_has_no_schema_setup_and_reads_existing_rows() {
    let (e, root) = seeded("ro-index", Arms::none());
    drop(e);
    let idx = Index::open_readonly(&root, true).unwrap();
    assert!(idx.is_read_only());
    assert!(idx.fts_ok());
    assert_eq!(idx.count(own().reader_scope()).unwrap(), 3);
    // The handle itself refuses a write (mode=ro plus query_only).
    let err = idx
        .connection(&own())
        .execute("DELETE FROM memories", [])
        .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("readonly")
            || err.to_string().contains("read-only")
            || err.to_string().contains("read only"),
        "{err}"
    );
}
