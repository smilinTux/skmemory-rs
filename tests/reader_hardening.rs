//! SKMEM-R-06D: reader hardening before the R-08 wiring.
//!
//! 1. Existence oracles: usage counters and counts, and the owner-only gate on
//!    everything that answers "does this id exist" for the whole store.
//! 2. `OwnerScope`: only an environment with no reader ceiling can mint one.
//! 3. Links: a visible memory never names a hidden one.
//!
//! Every test is hermetic (flat files and SQLite under a scratch dir).

use serde_json::json;
use skmem::audience::{AudienceConfig, NotOwner, OwnerScope, ReaderScope, READER_ENV};
use skmem::env::Env;
use skmem::flat::Store;
use skmem::index::{Index, SearchFilters};
use skmem::model::Memory;
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let base = std::env::var_os("SKMEM_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let d = base.join(format!("skmem-rs06d-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn own() -> OwnerScope {
    OwnerScope::resolve(&Env::empty()).unwrap()
}

fn scope(reader: &str) -> ReaderScope {
    ReaderScope::resolve(Some(reader), &Env::empty(), &AudienceConfig::empty())
}

fn mem(id: &str, ctx: &str, tags: &[&str]) -> Memory {
    let mut m = Memory::new(&format!("amberfox {id}"), &format!("amberfox river {id}"));
    m.id = id.into();
    m.context_tag = ctx.into();
    m.tags = tags.iter().map(|t| t.to_string()).collect();
    m
}

fn index_all(tag: &str, mems: &[Memory], fts: bool) -> (Store, Index, PathBuf) {
    let dir = tmp(tag);
    let store = Store::open(dir.join("m")).unwrap();
    let idx = Index::open(store.root(), fts).unwrap();
    for m in mems {
        store.save(m, &own()).unwrap();
        idx.index_memory(m, &store.path_for(m).unwrap(), &own())
            .unwrap();
    }
    (store, idx, dir)
}

fn usage_of(idx: &Index, id: &str) -> Option<(i64, Option<String>)> {
    idx.get_usage(&[id], &own().reader_scope().clone())
        .get(id)
        .cloned()
}

// ---------------------------------------------------------------- item 1

fn base() -> Vec<Memory> {
    vec![
        mem("pub1", "@public", &[]),
        mem("pub2", "@public", &[]),
        mem("secret", "@chef-only", &[]),
        mem("inner", "@inner-circle", &[]),
    ]
}

#[test]
fn bump_access_cannot_tell_hidden_from_missing_and_never_touches_hidden() {
    let (_s, idx, dir) = index_all("bump", &base(), true);
    let p = scope("@public");
    for ids in [["secret"], ["no-such-id"], ["inner"]] {
        assert_eq!(idx.bump_access(&ids, None, None, &p), 0, "{ids:?}");
    }
    // the hidden row's counter is untouched
    assert_eq!(usage_of(&idx, "secret").unwrap().0, 0);
    assert_eq!(usage_of(&idx, "inner").unwrap().0, 0);
    // a mixed batch counts only what the reader may see, and bumps only that
    assert_eq!(
        idx.bump_access(&["pub1", "secret", "ghost", "pub2"], None, None, &p),
        2
    );
    assert_eq!(usage_of(&idx, "pub1").unwrap().0, 1);
    assert_eq!(usage_of(&idx, "secret").unwrap().0, 0);
    // the owner is unchanged
    assert_eq!(
        idx.bump_access(&["secret", "ghost"], None, None, own().reader_scope()),
        1
    );
    assert_eq!(usage_of(&idx, "secret").unwrap().0, 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn get_usage_hides_hidden_ids_exactly_like_missing_ones() {
    let (_s, idx, dir) = index_all("usage", &base(), true);
    let p = scope("@public");
    let got = idx.get_usage(&["pub1", "secret", "ghost", "inner", ""], &p);
    assert_eq!(got.keys().cloned().collect::<Vec<_>>(), ["pub1"]);
    // an SQL-looking id is data, not syntax
    let evil = "x') OR 1=1 --";
    assert!(idx.get_usage(&[evil, "pub1"], &p).len() == 1);
    assert_eq!(idx.bump_access(&[evil], None, None, &p), 0);
    assert_eq!(idx.get_usage(&["secret"], own().reader_scope()).len(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn usage_calls_fail_closed_for_a_reader_when_fts_metadata_is_off() {
    let (_s, idx, dir) = index_all("usage-nofts", &base(), false);
    let p = scope("@public");
    assert_eq!(idx.bump_access(&["pub1"], None, None, &p), 0);
    assert!(idx.get_usage(&["pub1"], &p).is_empty());
    assert_eq!(idx.count(&p).unwrap(), 0);
    // the owner still works
    assert_eq!(idx.count(own().reader_scope()).unwrap(), 4);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn never_share_tags_are_a_hidden_id_for_usage_and_count() {
    let mut v = base();
    v.push(mem("rom", "@public", &["romantic"]));
    let (_s, idx, dir) = index_all("neversh", &v, true);
    let cfg = AudienceConfig::from_json(
        r#"{"channels":{"c":{"context_tag":"@public","members":["A"]}},
            "people":{"A":{"trust_level":1,"never_share":["romantic"]}}}"#,
    );
    let ch = ReaderScope::resolve(Some("channel:c"), &Env::empty(), &cfg);
    assert_eq!(idx.bump_access(&["rom"], None, None, &ch), 0);
    assert!(idx.get_usage(&["rom"], &ch).is_empty());
    assert_eq!(idx.count(&ch).unwrap(), 2);
    assert_eq!(usage_of(&idx, "rom").unwrap().0, 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn count_for_a_reader_never_reveals_the_store_total() {
    let (_s, idx, dir) = index_all("count", &base(), true);
    assert_eq!(idx.count(own().reader_scope()).unwrap(), 4);
    assert_eq!(idx.count(&scope("@public")).unwrap(), 2);
    assert_eq!(idx.count(&scope("@inner-circle")).unwrap(), 3);
    // an unknown or widening reader is public
    assert_eq!(idx.count(&scope("root")).unwrap(), 2);
    let ceiling = ReaderScope::resolve(
        Some("owner"),
        &Env::empty().with(READER_ENV, "@public"),
        &AudienceConfig::empty(),
    );
    assert_eq!(idx.count(&ceiling).unwrap(), 2);
    // count is the length of what the same reader can list
    let p = scope("@public");
    assert_eq!(
        idx.count(&p).unwrap() as usize,
        idx.all_rows(&p).unwrap().len()
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn owner_only_admin_calls_work_for_the_owner() {
    // fts_status, reindex, backfill, delete, archive, tombstone: typed to take an
    // OwnerScope; the compile_fail doctests in audience.rs pin the other half.
    let (store, idx, dir) = index_all("admin", &base(), true);
    let o = own();
    assert_eq!(idx.fts_status(&o).unwrap().unwrap().total, 4);
    assert!(idx
        .connection(&o)
        .query_row("SELECT 1", [], |r| r.get::<_, i64>(0))
        .is_ok());
    store
        .tombstone("pub1", "2026-11-07T00:00:00+0000", "t", None, &o)
        .unwrap();
    assert!(store.is_tombstoned("pub1", &o));
    assert!(!store.is_tombstoned("never-seen", &o));
    assert!(store.archive("pub2", None, &o).unwrap().is_some());
    assert!(store.archive("pub2", None, &o).unwrap().is_none());
    assert!(!store.delete("pub2", &o).unwrap());
    assert!(idx.delete("pub1", &o).unwrap());
    std::fs::remove_dir_all(dir).unwrap();
}

// ---------------------------------------------------------------- item 2

#[test]
fn owner_scope_is_only_minted_for_an_owner_environment() {
    let ok = |v: &str| OwnerScope::resolve(&Env::empty().with(READER_ENV, v)).is_ok();
    // nothing, blank, and the owner spellings (case and spaces as Python)
    assert!(OwnerScope::resolve(&Env::empty()).is_ok());
    for v in [
        "",
        "   ",
        "owner",
        "@chef",
        "@chef-only",
        "  @CHEF-ONLY ",
        "OWNER",
    ] {
        assert!(ok(v), "{v:?} should be the owner");
    }
    // every ceiling that is not the owner is refused, however it is spelled
    for v in [
        "@public",
        "@community",
        "@work-circle",
        "@inner-circle",
        "@work:a",
        "@inner:x",
        "channel:c",
        "channel:",
        "root",
        "admin",
        "*",
        "@everyone",
        "@chef-only-ish",
        "owner2",
        "\u{200b}owner",
    ] {
        match OwnerScope::resolve(&Env::empty().with(READER_ENV, v)) {
            Err(NotOwner { declared }) => assert_eq!(declared, v.trim()),
            Ok(_) => panic!("{v:?} minted an OwnerScope"),
        }
    }
    let msg = OwnerScope::resolve(&Env::empty().with(READER_ENV, "@public"))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("refused") && msg.contains(READER_ENV));
}

#[test]
fn owner_scope_gives_the_unrestricted_read_scope_and_nothing_wider() {
    let o = own();
    assert!(o.reader_scope().unrestricted());
    // narrowing the owner scope still narrows
    let n = o.reader_scope().narrow(&scope("@public"));
    assert!(!n.unrestricted());
    assert!(!n.allows(Some("@chef-only"), &[]));
}

// ---------------------------------------------------------------- item 3

fn linked() -> Vec<Memory> {
    let mut v = mem("v", "@public", &[]);
    v.related_ids = vec![
        "pub2".into(),
        "secret".into(),
        "ghost".into(),
        "../etc/passwd".into(),
        "rom".into(),
        "pub2".into(),
    ];
    v.parent_id = Some("secret".into());
    v.supersedes_id = Some("secret".into());
    v.metadata.insert(
        "superseded_by".into(),
        json!({"id": "secret", "reason": "edit", "at": "2026-01-01T00:00:00+00:00"}),
    );
    // a second visible memory whose links are all visible
    let mut w = mem("w", "@public", &[]);
    w.related_ids = vec!["pub2".into(), "v".into()];
    w.parent_id = Some("pub2".into());
    w.supersedes_id = Some("pub2".into());
    w.metadata
        .insert("superseded_by".into(), json!({"id": "v", "reason": "r"}));
    // a link to a malformed superseded_by value
    let mut x = mem("x", "@public", &[]);
    x.metadata.insert("superseded_by".into(), json!("secret"));
    vec![
        v,
        w,
        x,
        mem("pub2", "@public", &[]),
        mem("secret", "@chef-only", &[]),
        mem("rom", "@public", &["romantic"]),
    ]
}

fn channel_no_romantic() -> ReaderScope {
    let cfg = AudienceConfig::from_json(
        r#"{"channels":{"c":{"context_tag":"@public","members":["A"]}},
            "people":{"A":{"trust_level":1,"never_share":["romantic"]}}}"#,
    );
    ReaderScope::resolve(Some("channel:c"), &Env::empty(), &cfg)
}

fn check_redacted(m: &Memory, related: &[&str]) {
    assert_eq!(m.related_ids, related, "only visible ids survive");
    assert_eq!(m.parent_id, None);
    assert_eq!(m.supersedes_id, None);
    assert!(
        !m.metadata.contains_key("superseded_by"),
        "a hidden successor must not leave a trace: {:?}",
        m.metadata
    );
    // nothing else changed
    assert_eq!(m.title, "amberfox v");
}

#[test]
fn flat_load_and_list_redact_links_to_hidden_dangling_and_unsafe_ids() {
    let (store, _i, dir) = index_all("flat-links", &linked(), true);
    // "rom" is @public but tagged romantic: visible to @public, never-shared to the channel
    for (sc, rel) in [
        (scope("@public"), vec!["pub2", "rom", "pub2"]),
        (channel_no_romantic(), vec!["pub2", "pub2"]),
    ] {
        let m = store.load("v", &sc).unwrap().unwrap();
        check_redacted(&m, &rel);
        let listed = store.list(None, &sc);
        check_redacted(listed.memories.iter().find(|m| m.id == "v").unwrap(), &rel);
    }
    // visible links survive
    let w = store.load("w", &scope("@public")).unwrap().unwrap();
    assert_eq!(w.related_ids, ["pub2", "v"]);
    assert_eq!(w.parent_id.as_deref(), Some("pub2"));
    assert_eq!(w.supersedes_id.as_deref(), Some("pub2"));
    assert_eq!(w.metadata["superseded_by"]["id"], "v");
    // a malformed superseded_by cannot be vetted: dropped for a reader
    let x = store.load("x", &scope("@public")).unwrap().unwrap();
    assert!(!x.metadata.contains_key("superseded_by"));
    // the owner sees everything, byte for byte
    let own_v = store.load("v", own().reader_scope()).unwrap().unwrap();
    assert_eq!(own_v.related_ids.len(), 6);
    assert_eq!(own_v.parent_id.as_deref(), Some("secret"));
    assert_eq!(own_v.metadata["superseded_by"]["id"], "secret");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_hidden_link_and_a_missing_link_are_indistinguishable() {
    // two stores: v links to a HIDDEN id in one and to a MISSING id in the other
    let mut hidden = vec![mem("v", "@public", &[]), mem("t", "@chef-only", &[])];
    let mut missing = vec![mem("v", "@public", &[])];
    for set in [&mut hidden, &mut missing] {
        let v = &mut set[0];
        v.related_ids = vec!["t".into()];
        v.parent_id = Some("t".into());
        v.supersedes_id = Some("t".into());
        v.metadata
            .insert("superseded_by".into(), json!({"id": "t"}));
        let fixed = "2026-01-01T00:00:00+00:00";
        v.created_at = fixed.into();
        v.updated_at = fixed.into();
        v.event_at = Some(fixed.into());
        v.valid_from = Some(fixed.into());
    }
    let (sa, ia, da) = index_all("ind-a", &hidden, true);
    let (sb, ib, db) = index_all("ind-b", &missing, true);
    let p = scope("@public");
    assert_eq!(
        sa.load("v", &p).unwrap().unwrap().to_json(),
        sb.load("v", &p).unwrap().unwrap().to_json()
    );
    let strip = |mut r: skmem::index::MemoryRow| {
        r.file_path.clear();
        r
    };
    assert_eq!(
        ia.get("v", &p).unwrap().map(strip),
        ib.get("v", &p).unwrap().map(strip)
    );
    let rows = |i: &Index| -> Vec<_> { i.all_rows(&p).unwrap().into_iter().map(strip).collect() };
    assert_eq!(rows(&ia), rows(&ib));
    std::fs::remove_dir_all(da).unwrap();
    std::fs::remove_dir_all(db).unwrap();
}

#[test]
fn index_rows_redact_links_on_get_all_rows_and_search() {
    let (_s, idx, dir) = index_all("idx-links", &linked(), true);
    for (sc, rel) in [
        (scope("@public"), "pub2,rom,pub2"),
        (channel_no_romantic(), "pub2,pub2"),
    ] {
        let row = idx.get("v", &sc).unwrap().unwrap();
        assert_eq!(row.related_ids, rel);
        assert_eq!(row.parent_id, None);
        assert_eq!(row.supersedes_id, None);
        let all = idx.all_rows(&sc).unwrap();
        let v = all.iter().find(|r| r.id == "v").unwrap();
        assert_eq!(v.related_ids, rel);
        assert_eq!(v.parent_id, None);
        let hits = idx
            .search_lexical("amberfox", 50, &SearchFilters::default(), &sc)
            .unwrap();
        let v = hits.iter().find(|r| r.id == "v").unwrap();
        assert_eq!(v.related_ids, rel);
        assert_eq!(v.parent_id, None);
        assert_eq!(v.supersedes_id, None);
        // visible neighbours stay
        let w = hits.iter().find(|r| r.id == "w").unwrap();
        assert_eq!(w.related_ids, "pub2,v");
        assert_eq!(w.parent_id.as_deref(), Some("pub2"));
    }
    // owner: unchanged
    let row = idx.get("v", own().reader_scope()).unwrap().unwrap();
    assert!(row.related_ids.contains("secret") && row.related_ids.contains("../etc/passwd"));
    assert_eq!(row.parent_id.as_deref(), Some("secret"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn redaction_cannot_be_widened_by_a_per_call_owner_under_an_env_ceiling() {
    let (store, idx, dir) = index_all("ceiling", &linked(), true);
    let env = Env::empty().with(READER_ENV, "@public");
    let widened = ReaderScope::resolve(Some("owner"), &env, &AudienceConfig::empty());
    check_redacted(
        &store.load("v", &widened).unwrap().unwrap(),
        &["pub2", "rom", "pub2"],
    );
    assert_eq!(
        idx.get("v", &widened).unwrap().unwrap().related_ids,
        "pub2,rom,pub2"
    );
    // and a bare redaction on an owner-resolved scope is a no-op
    let mut m = linked().remove(0);
    ReaderScope::resolve(None, &Env::empty(), &AudienceConfig::empty())
        .redact_links(&mut m, &mut |_| false);
    assert_eq!(m.related_ids.len(), 6);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn redact_links_fails_closed_when_the_oracle_says_no() {
    let mut m = linked().remove(0);
    scope("@public").redact_links(&mut m, &mut |_| false);
    assert!(m.related_ids.is_empty());
    assert!(m.parent_id.is_none() && m.supersedes_id.is_none());
    assert!(!m.metadata.contains_key("superseded_by"));
    let mut m = linked().remove(0);
    scope("@public").redact_links(&mut m, &mut |_| true);
    assert_eq!(m.related_ids.len(), 6);
}
