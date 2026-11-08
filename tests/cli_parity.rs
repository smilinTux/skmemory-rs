//! SKMEM-R-10: the `skmem` CLI against the real Python CLI.
//!
//! `tests/cli_parity/expected.json` is written by `tests/cli_parity/gen_cli_parity.py`
//! from the Python `skmemory.cli` over the golden corpus store. Every case here
//! runs the `skmem` binary in a fresh copy of the same store (after a `reindex`,
//! as the generator does) with the same `SKMEMORY_READER` and must print the same
//! bytes and exit with the same status. The other tests pin what has no Python
//! twin (`--json`, `intentions list`, the owner-only refusals, store resolution).
//! Hermetic: no Python, no network, no Postgres.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// A scratch home holding `agents/fx/memory` built like the generator builds it.
struct Home {
    dir: PathBuf,
}

impl Home {
    fn new(tag: &str) -> Home {
        let dir = std::env::temp_dir().join(format!("skmem-cli-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&dir);
        let mem = dir.join("agents/fx/memory");
        for f in fs::read_dir(root().join("golden/out")).unwrap() {
            let f = f.unwrap().path();
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            if !(name.starts_with("corpus-") || name == "full-v2-unicode.json") {
                continue;
            }
            let text = fs::read_to_string(&f).unwrap();
            let d: Value = serde_json::from_str(&text).unwrap();
            let tier = mem.join(d["layer"].as_str().unwrap());
            fs::create_dir_all(&tier).unwrap();
            fs::write(
                tier.join(format!("{}.json", d["id"].as_str().unwrap())),
                text,
            )
            .unwrap();
        }
        copy_dir(
            &root().join("intent_parity/py_store/intentions"),
            &mem.join("intentions"),
        );
        copy_dir(
            &root().join("intent_parity/py_ledger/ledger"),
            &mem.join("ledger"),
        );
        Home { dir }
    }

    fn mem(&self) -> PathBuf {
        self.dir.join("agents/fx/memory")
    }

    fn cmd(&self, reader: Option<&str>, args: &[String]) -> Output {
        let mut c = Command::new(env!("CARGO_BIN_EXE_skmem"));
        c.env_clear()
            .env("HOME", &self.dir)
            .env("SKCAPSTONE_HOME", &self.dir)
            .env("SKAGENT", "fx")
            .env("SKCAPSTONE_AGENT", "fx")
            .env("SKMEMORY_AGENT", "fx");
        if let Some(r) = reader {
            c.env("SKMEMORY_READER", r);
        }
        c.args(args).output().expect("run skmem")
    }

    fn run(&self, reader: Option<&str>, args: &[&str]) -> Output {
        let v: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        self.cmd(reader, &v)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for e in fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let to = dst.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &to);
        } else {
            fs::copy(e.path(), to).unwrap();
        }
    }
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A `list` page as (header, sorted record blocks): ties on `created_at` come back in
/// an order Python leaves to SQLite row order, so the page is compared as a set.
fn blocks(s: &str) -> (String, Vec<String>) {
    let mut parts = s.split("\n\n");
    let head = parts.next().unwrap_or("").to_string();
    let mut rest: Vec<String> = parts.map(str::to_string).collect();
    rest.sort();
    (head, rest)
}

#[test]
fn every_case_matches_the_python_cli() {
    let expected: Value =
        serde_json::from_str(&fs::read_to_string(root().join("cli_parity/expected.json")).unwrap())
            .unwrap();
    let cases = expected.as_object().unwrap();
    assert!(cases.len() >= 30, "expected.json lost cases");
    let mut bad = Vec::new();
    for (name, c) in cases {
        let home = Home::new(name);
        let mem = home.mem().to_string_lossy().into_owned();
        let pre = home.run(None, &["reindex"]);
        assert_eq!(pre.status.code(), Some(0), "{name} reindex: {}", err(&pre));
        let args: Vec<String> = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().replace("{MEM}", &mem))
            .collect();
        let o = home.cmd(c["reader"].as_str(), &args);
        let want = c["stdout"]
            .as_str()
            .unwrap()
            .replace("{HOME}", &home.dir.to_string_lossy());
        let want_code = c["code"].as_i64().unwrap() as i32;
        let same = if name.starts_with("list-") {
            blocks(&out(&o)) == blocks(&want)
        } else {
            out(&o) == want
        };
        if !same || o.status.code() != Some(want_code) {
            bad.push(format!(
                "{name}: code {:?} (want {want_code})\n--- got\n{}\n--- want\n{want}\n--- stderr\n{}",
                o.status.code(),
                out(&o),
                err(&o)
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} case(s) differ:\n{}",
        bad.len(),
        bad.join("\n====\n")
    );
}

#[test]
fn bare_binary_still_prints_the_version() {
    let o = Command::new(env!("CARGO_BIN_EXE_skmem")).output().unwrap();
    assert_eq!(out(&o).trim(), env!("CARGO_PKG_VERSION"));
    let v = Command::new(env!("CARGO_BIN_EXE_skmem"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(out(&v).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn search_json_has_the_binding_shape_and_filters_apply() {
    let h = Home::new("json");
    h.run(None, &["reindex"]);
    let o = h.run(None, &["search", "embedding model", "--json"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let rows: Value = serde_json::from_str(&out(&o)).unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let old = rows.iter().find(|r| r["id"] == "m-06159e9b16").unwrap();
    assert_eq!(old["superseded"], Value::Bool(true));
    assert_eq!(old["superseded_by"], Value::String("m-794ef9bd00".into()));
    assert!(old["content"].is_string() && old["emotional"].is_object());
    // layer and tag filters (no Python twin): the long-term unicode record only.
    let o = h.run(
        None,
        &["search", "quoted", "--layer", "long-term", "--json"],
    );
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    assert!(v
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["layer"] == "long-term"));
    let o = h.run(None, &["search", "standup", "--tags", "standup", "--json"]);
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    assert!(!v.as_array().unwrap().is_empty());
    assert!(v
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["tags"].as_array().unwrap().contains(&"standup".into())));
    let o = h.run(
        None,
        &["search", "standup", "--layer", "short-term", "--json"],
    );
    assert_eq!(out(&o).trim(), "[]");
}

#[test]
fn list_is_newest_first() {
    let h = Home::new("order");
    let o = h.run(None, &["list", "--limit", "100", "--json"]);
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    let times: Vec<&str> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["created_at"].as_str().unwrap())
        .collect();
    assert_eq!(times.len(), 60);
    assert!(times.windows(2).all(|w| w[0] >= w[1]), "not newest first");
}

#[test]
fn list_json_filters_by_layer_and_tag() {
    let h = Home::new("listjson");
    let o = h.run(
        None,
        &[
            "list", "--layer", "mid-term", "--tags", "rollout", "--limit", "3", "--json",
        ],
    );
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    let rows = v.as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r["layer"] == "mid-term"));
}

#[test]
fn a_reader_ceiling_hides_private_rows_everywhere() {
    let h = Home::new("ceiling");
    h.run(None, &["reindex"]);
    for args in [
        vec!["search", "medication timing", "--json"],
        vec!["list", "--limit", "100", "--json"],
    ] {
        let o = h.run(Some("@public"), &args);
        assert!(
            !out(&o).contains("m-2ae3e47c3c"),
            "{args:?} leaked a private id"
        );
        assert!(
            !out(&o).contains("medication"),
            "{args:?} leaked private text"
        );
    }
    // --reader narrows an owner process, never the reverse.
    let o = h.run(
        None,
        &[
            "search",
            "medication timing",
            "--reader",
            "@public",
            "--json",
        ],
    );
    assert_eq!(out(&o).trim(), "[]");
    let o = h.run(
        Some("@public"),
        &[
            "search",
            "medication timing",
            "--reader",
            "@chef-only",
            "--json",
        ],
    );
    assert_eq!(out(&o).trim(), "[]");
}

#[test]
fn owner_only_commands_refuse_under_a_reader_and_write_nothing() {
    let h = Home::new("owneronly");
    let before = fs::read_dir(h.mem().join("mid-term")).unwrap().count();
    for args in [
        vec!["snapshot", "T", "C"],
        vec!["reindex"],
        vec!["intentions", "list"],
        vec!["intentions", "reindex"],
        vec!["intentions", "done", "int-001"],
        vec!["ledger", "top"],
    ] {
        let o = h.run(Some("@public"), &args);
        assert_eq!(o.status.code(), Some(1), "{args:?}");
        assert!(err(&o).contains("owner-only"), "{args:?}: {}", err(&o));
        assert!(out(&o).is_empty());
    }
    assert_eq!(
        fs::read_dir(h.mem().join("mid-term")).unwrap().count(),
        before
    );
    assert!(
        !h.mem().join("short-term").exists(),
        "a refused snapshot created a tier"
    );
}

#[test]
fn health_reports_the_engine_and_a_reader_gets_only_the_verdict() {
    let h = Home::new("health");
    let o = h.run(None, &["health"]);
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    assert_eq!(v["engine"], "skmem-rust");
    assert_eq!(
        v["flat_only"], 60,
        "unindexed store: all 60 files are flat-only"
    );
    assert_eq!(v["status"], "degraded");
    h.run(None, &["reindex"]);
    let v: Value = serde_json::from_str(&out(&h.run(None, &["health"]))).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["primary"]["total_memories"], 60);
    assert_eq!(v["primary"]["sync"]["in_sync"], true);
    let r: Value = serde_json::from_str(&out(&h.run(Some("@public"), &["health"]))).unwrap();
    assert!(r.get("primary").is_none() && r.get("flat_only").is_none());
    assert!(r["status"].is_string());
}

#[test]
fn snapshot_then_recall_roundtrips_and_dedups() {
    let h = Home::new("snap");
    let o = h.run(
        None,
        &[
            "snapshot",
            "Cli title",
            "cli body about zebras",
            "--layer",
            "long-term",
            "--tags",
            "a, b",
            "--emotions",
            "joy,trust",
            "--intensity",
            "8.5",
            "--valence",
            "0.9",
            "--resonance",
            "bright",
        ],
    );
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let text = out(&o);
    let id = text
        .lines()
        .next()
        .unwrap()
        .strip_prefix("Snapshot saved: ")
        .unwrap()
        .to_string();
    assert!(
        text.contains("  Layer: long-term\n  Emotional: joy+trust@8.5/+0.9\n"),
        "{text}"
    );
    let r: Value = serde_json::from_str(&out(&h.run(None, &["recall", &id]))).unwrap();
    assert_eq!(r["title"], "Cli title");
    assert_eq!(r["source"], "cli");
    assert_eq!(r["tags"], serde_json::json!(["a", "b"]));
    assert!(!r["integrity_hash"].as_str().unwrap().is_empty());
    // it is searchable straight away, and a second identical snapshot is a no-op.
    assert!(out(&h.run(None, &["search", "zebras"])).contains("Cli title"));
    let again = out(&h.run(None, &["snapshot", "Cli title", "cli body about zebras"]));
    assert!(again.contains(&id), "{again}");
    // validation and the decompose refusal write nothing.
    let bad = h.run(None, &["snapshot", "t", "c", "--intensity", "11"]);
    assert_eq!(bad.status.code(), Some(1));
    let big = h.run(None, &["snapshot", "t", &"x".repeat(1300)]);
    assert_eq!(big.status.code(), Some(1));
    assert!(err(&big).contains("decompose"));
    let usage = h.run(None, &["snapshot", "t", "c", "--layer", "nope"]);
    assert_eq!(usage.status.code(), Some(2));
}

#[test]
fn intentions_list_and_done() {
    let h = Home::new("intent");
    // the index is derived: empty until `intentions reindex` reads the synced flat files
    let empty = h.run(None, &["intentions", "list"]);
    assert_eq!(out(&empty).trim(), "[]");
    let r = h.run(None, &["intentions", "reindex"]);
    assert_eq!(r.status.code(), Some(0), "{}", err(&r));
    assert!(out(&r).starts_with("Indexed "), "{}", out(&r));
    let o = h.run(None, &["intentions", "list"]);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let all: Value = serde_json::from_str(&out(&o)).unwrap();
    let n = all.as_array().unwrap().len();
    assert!(n >= 12, "{n}");
    let ids: Vec<&str> = all
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["id"].as_str().unwrap())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "ordered by id like IntentionStore.list");
    let first = ids[0].to_string();
    let state = all[0]["state"].as_str().unwrap().to_string();
    let by_state: Value = serde_json::from_str(&out(
        &h.run(None, &["intentions", "list", "--state", &state])
    ))
    .unwrap();
    assert!(by_state
        .as_array()
        .unwrap()
        .iter()
        .all(|i| i["state"] == state.as_str()));
    let d = h.run(None, &["intentions", "done", &first]);
    assert_eq!(d.status.code(), Some(0), "{}", err(&d));
    assert_eq!(out(&d).trim(), format!("{first}: done"));
    let missing = h.run(None, &["intentions", "done", "no-such-id"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(err(&missing).contains("intention not found: no-such-id"));
}

#[test]
fn store_resolution_refuses_to_guess() {
    let h = Home::new("resolve");
    let run = |envs: &[(&str, &str)]| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_skmem"));
        c.env_clear()
            .env("HOME", &h.dir)
            .env("SKCAPSTONE_HOME", &h.dir);
        for (k, v) in envs {
            c.env(k, v);
        }
        c.args(["list"]).output().unwrap()
    };
    // a sole installed agent is unambiguous
    assert_eq!(run(&[]).status.code(), Some(0));
    // conflicting selectors refuse
    let o = run(&[("SKAGENT", "fx"), ("SKMEMORY_AGENT", "other")]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("No active agent memory profile"),
        "{}",
        err(&o)
    );
    // two agents and no declaration refuse
    fs::create_dir_all(h.dir.join("agents/second/memory")).unwrap();
    let o = run(&[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("without guessing"), "{}", err(&o));
    // SK_DEFAULT_AGENT resolves it; --base overrides everything
    assert_eq!(run(&[("SK_DEFAULT_AGENT", "fx")]).status.code(), Some(0));
    assert!(out(&run(&[("SK_DEFAULT_AGENT", "second")])).contains("No memories found."));
    let o = h.run(
        None,
        &["--base", &h.mem().to_string_lossy(), "list", "--limit", "1"],
    );
    assert!(out(&o).starts_with("Found 1 memories:"));
}
