//! Flat store behaviour: layout, conflict copies, archive, and crash safety.

use skmem::flat::{atomic_write_bytes, atomic_write_crashing, CrashPoint, Store};
use skmem::model::{Layer, Memory};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skmem-flat-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn mem(id: &str, layer: Layer, created: &str) -> Memory {
    let mut m = Memory::new("title", "content");
    m.id = id.into();
    m.layer = layer;
    m.created_at = created.into();
    m.updated_at = created.into();
    m.event_at = Some(created.into());
    m.valid_from = Some(created.into());
    m
}

#[test]
fn save_load_list_roundtrip_and_layout() {
    let d = scratch("rt");
    let s = Store::open(&d).unwrap();
    for l in Layer::ALL {
        assert!(d.join(l.as_str()).is_dir());
    }
    let a = mem("a", Layer::Short, "2026-01-01T00:00:00+00:00");
    let b = mem("b", Layer::Long, "2026-02-01T00:00:00+00:00");
    s.save(&a).unwrap();
    s.save(&b).unwrap();
    assert!(d.join("short-term/a.json").is_file());
    assert_eq!(s.load("a").unwrap().unwrap(), a);
    assert_eq!(s.load("b").unwrap().unwrap(), b);
    assert!(s.load("nope").unwrap().is_none());
    let ids: Vec<String> = s.list(None).into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["b", "a"], "newest first");
    assert_eq!(s.list(Some(Layer::Short)).len(), 1);
    // The saved bytes are exactly the Python dump and carry no trailing newline.
    let raw = fs::read_to_string(d.join("short-term/a.json")).unwrap();
    assert_eq!(raw, a.to_json());
    assert!(!raw.ends_with('\n'));
    // No temp litter after a clean save.
    assert!(fs::read_dir(d.join("short-term")).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with('.')));
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn unsafe_ids_are_refused() {
    let d = scratch("ids");
    let s = Store::open(&d).unwrap();
    for bad in ["", "  ", "../x", "a/b", ".."] {
        let mut m = mem("x", Layer::Short, "2026-01-01T00:00:00+00:00");
        m.id = bad.into();
        assert!(s.save(&m).is_err(), "{bad:?}");
    }
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn invalid_files_are_skipped_by_list_and_reported_by_load() {
    let d = scratch("invalid");
    let s = Store::open(&d).unwrap();
    fs::write(d.join("short-term/bad.json"), b"{not json").unwrap();
    fs::write(d.join("short-term/.hidden.json"), b"{}").unwrap();
    s.save(&mem("ok", Layer::Short, "2026-01-01T00:00:00+00:00"))
        .unwrap();
    assert_eq!(s.list(None).len(), 1);
    assert!(s.load("bad").is_err());
    assert_eq!(s.flat_id_set().len(), 2, "dotfiles are not memories");
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn sync_conflict_copies_are_skipped_unless_only_copy() {
    let d = scratch("conflict");
    let s = Store::open(&d).unwrap();
    let canon = mem("c1", Layer::Mid, "2026-01-01T00:00:00+00:00");
    s.save(&canon).unwrap();
    // A divergent conflict copy of c1 sits next to the canonical file.
    let mut div = canon.clone();
    div.title = "divergent".into();
    fs::write(
        d.join("mid-term/c1.sync-conflict-20260102-101010-ABCDEFG.json"),
        div.to_json(),
    )
    .unwrap();
    // c2 exists only as two conflict copies; the newest (sorted last) wins.
    let mut only = mem("c2", Layer::Long, "2026-03-01T00:00:00+00:00");
    only.title = "older".into();
    fs::write(
        d.join("long-term/c2.sync-conflict-20260301-000000-AAAAAAA.json"),
        only.to_json(),
    )
    .unwrap();
    only.title = "newer".into();
    fs::write(
        d.join("long-term/c2.sync-conflict-20260302-000000-BBBBBBB.json"),
        only.to_json(),
    )
    .unwrap();

    let listed = s.list(None);
    let titles: Vec<&str> = listed.iter().map(|m| m.title.as_str()).collect();
    assert_eq!(listed.len(), 2, "{titles:?}");
    assert!(titles.contains(&"title"), "canonical copy listed");
    assert!(
        titles.contains(&"newer"),
        "only-copy conflict listed, newest"
    );
    assert!(!titles.contains(&"divergent") && !titles.contains(&"older"));
    assert_eq!(s.load("c1").unwrap().unwrap().title, "title");
    assert_eq!(s.load("c2").unwrap().unwrap().title, "newer");
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn tombstone_and_archive() {
    let d = scratch("arch");
    let s = Store::open(&d).unwrap();
    let m = mem("gone", Layer::Short, "2026-01-01T00:00:00+00:00");
    s.save(&m).unwrap();
    let before = fs::read(d.join("short-term/gone.json")).unwrap();
    let dest = s.archive("gone", Some("deduped")).unwrap().unwrap();
    assert_eq!(dest, d.join("archive/deduped/gone.json"));
    assert_eq!(
        fs::read(&dest).unwrap(),
        before,
        "bytes untouched by archive"
    );
    assert!(s.load("gone").unwrap().is_none());
    assert!(s.archive("gone", None).unwrap().is_none());
    s.tombstone("gone", "2026-11-07T00:00:00+0000", "forget", None)
        .unwrap();
    assert!(s.is_tombstoned("gone"));
    let m2 = mem("plain", Layer::Mid, "2026-01-01T00:00:00+00:00");
    s.save(&m2).unwrap();
    let p = s.archive("plain", None).unwrap().unwrap();
    assert_eq!(p, d.join("archive/plain.json"));
    assert!(!s.delete("plain").unwrap());
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn overwrite_replaces_whole_file() {
    let d = scratch("over");
    let s = Store::open(&d).unwrap();
    let mut m = mem("o", Layer::Short, "2026-01-01T00:00:00+00:00");
    m.content = "x".repeat(5000);
    s.save(&m).unwrap();
    m.content = "short".into();
    s.save(&m).unwrap();
    assert_eq!(s.load("o").unwrap().unwrap().content, "short");
    let _ = fs::remove_dir_all(&d);
}

// ---- crash safety --------------------------------------------------------

#[test]
fn crash_before_rename_leaves_prior_file_intact_and_temp_invisible() {
    for point in [CrashPoint::AfterTempWrite, CrashPoint::AfterFsync] {
        let d = scratch("crash");
        let s = Store::open(&d).unwrap();
        let mut m = mem("k", Layer::Short, "2026-01-01T00:00:00+00:00");
        s.save(&m).unwrap();
        let before = fs::read(d.join("short-term/k.json")).unwrap();

        m.content = "NEW CONTENT".repeat(1000);
        let path = s.path_for(&m).unwrap();
        let r = atomic_write_crashing(&path, m.to_json().as_bytes(), point);
        assert!(r.is_err());

        // The target is exactly the old bytes; the orphaned temp is not a memory.
        assert_eq!(fs::read(&path).unwrap(), before, "{point:?}");
        let temps = fs::read_dir(d.join("short-term"))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp-")
            })
            .count();
        assert_eq!(temps, 1, "crash leaves the temp behind, as a kill would");
        assert_eq!(s.list(None).len(), 1);
        assert_eq!(s.flat_id_set().len(), 1);
        assert_eq!(s.load("k").unwrap().unwrap().content, "content");
        let _ = fs::remove_dir_all(&d);
    }
}

#[test]
fn crash_before_first_write_leaves_no_target() {
    let d = scratch("crash-new");
    let s = Store::open(&d).unwrap();
    let m = mem("n", Layer::Short, "2026-01-01T00:00:00+00:00");
    let path = s.path_for(&m).unwrap();
    assert!(atomic_write_crashing(&path, b"{}", CrashPoint::AfterFsync).is_err());
    assert!(!path.exists());
    assert!(s.load("n").unwrap().is_none());
    let _ = fs::remove_dir_all(&d);
}

#[test]
fn failed_write_cleans_up_its_temp() {
    let d = scratch("fail");
    // Renaming a file over a directory fails; the temp must not be left behind.
    let target = d.join("target.json");
    fs::create_dir_all(target.join("child")).unwrap();
    assert!(atomic_write_bytes(&target, b"data").is_err());
    let leftovers = fs::read_dir(&d)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        })
        .count();
    assert_eq!(leftovers, 0);
    let _ = fs::remove_dir_all(&d);
}

/// Child half of the kill test: save two alternating large payloads forever.
#[test]
fn kill_child_writer() {
    let Ok(dir) = std::env::var("SKMEM_KILL_CHILD_DIR") else {
        return;
    };
    let s = Store::open(&dir).unwrap();
    let mut i = 0u64;
    loop {
        let mut m = mem("victim", Layer::Short, "2026-01-01T00:00:00+00:00");
        m.content = if i % 2 == 0 { "A" } else { "B" }.repeat(400_000);
        s.save(&m).unwrap();
        i += 1;
    }
}

/// Parent half: SIGKILL a process mid-save several times; the target must
/// always be a complete, valid memory holding one of the two payloads.
#[test]
fn sigkill_mid_save_never_leaves_a_partial_file() {
    let d = scratch("kill");
    let exe = std::env::current_exe().unwrap();
    for round in 0..6u64 {
        let mut child = Command::new(&exe)
            .args(["kill_child_writer", "--exact", "--nocapture"])
            .env("SKMEM_KILL_CHILD_DIR", &d)
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(80 + round * 37));
        let _ = child.kill();
        let _ = child.wait();
        let s = Store::open_readonly(&d);
        if let Some(m) = s.load("victim").unwrap() {
            let c = &m.content;
            assert_eq!(c.len(), 400_000, "round {round}: truncated file");
            assert!(
                c.bytes().all(|b| b == c.as_bytes()[0]),
                "round {round}: mixed"
            );
        }
        // Any temp a kill orphaned is invisible to list/load.
        assert!(s.list(None).len() <= 1);
    }
    let _ = fs::remove_dir_all(&d);
}
