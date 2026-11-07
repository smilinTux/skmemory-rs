//! SKMEM-R-07: the Python eval fixtures as the conformance suite, plus the
//! semantics the Python tests pin (locks, try-lock skip, terminal rules,
//! caps, deadline, tick delivery). Hermetic.

mod intent_common;

use intent_common::*;
use serde_json::{json, Value};
use skmem::intentions::tick::{run_tick, sk_alert_deliver, TickOptions};
use skmem::intentions::{
    lock_dir_for, prefilter_safe, transition, Intention, IntentionStore, Kind, PrefilterOpts, State,
};
use skmem::ledger::{self, Ledger, Record};
use skmem::pytime::Dt;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn dt(s: &str) -> Dt {
    Dt::parse(s).unwrap()
}

fn open(base: &Path) -> IntentionStore {
    IntentionStore::open_with(base, 30_000, Some(&base.join("..").join("locks")), &own()).unwrap()
}

fn none() -> HashMap<String, String> {
    HashMap::new()
}

/// A store of the fixture intentions plus their origin memories' `valid_to`.
fn fixture_store(tag: &str) -> (PathBuf, IntentionStore) {
    let base = scratch(tag).join("base");
    let st = open(&base);
    for row in jsonl(&fixtures().join("intentions-v1.jsonl")) {
        st.save(&Intention::from_value(&row).unwrap()).unwrap();
    }
    let lt = base.join("long-term");
    fs::create_dir_all(&lt).unwrap();
    for r in jsonl(&fixtures().join("corpus-valid-to-v1.jsonl")) {
        fs::write(
            lt.join(format!("{}.json", r["id"].as_str().unwrap())),
            r.to_string(),
        )
        .unwrap();
    }
    (base, st)
}

#[test]
fn transition_fixture_rows() {
    let rows = jsonl(&fixtures().join("intention-transitions-v1.jsonl"));
    assert_eq!(rows.len(), 16);
    for r in rows {
        let mut i = Intention::new("x", Kind::Time, "t");
        i.state = State::parse(r["state"].as_str().unwrap()).unwrap();
        let out = transition(
            &i,
            r["event"].as_str().unwrap(),
            Some(dt("2026-11-01T12:00:00Z")),
            None,
        )
        .unwrap();
        assert_eq!(out.state.as_str(), r["expected"].as_str().unwrap(), "{r}");
    }
}

#[test]
fn prompt_fixture_vectors_fire_exactly_the_expected_ids() {
    let prompts = jsonl(&fixtures().join("prompts-v1.jsonl"));
    assert_eq!(prompts.len(), 17);
    for p in prompts {
        let now = dt(p["now"].as_str().unwrap());
        let (_b, st) = fixture_store("prompt");
        let id = p["id"].as_str().unwrap();
        let want: Vec<String> = p["expected_fire"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let got: Vec<String> = if id == "p-17" {
            // the time tick, not the prefilter
            let mut ok = |_: &str| true;
            run_tick(
                &st,
                TickOptions {
                    now: Some(now),
                    deliver: Some(&mut ok),
                    any_host: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .fired
        } else {
            st.sweep_supersession(Some(now)).unwrap();
            let scope: HashMap<String, String> = p["scope"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect();
            st.prefilter(
                p["text"].as_str().unwrap(),
                &scope,
                Some(now),
                &PrefilterOpts::default(),
            )
            .unwrap()
            .into_iter()
            .map(|i| i.id)
            .collect()
        };
        assert_eq!(got, want, "{id}: {}", p["text"]);
    }
}

#[test]
fn a_fire_updates_flat_file_and_index_together() {
    let (base, st) = fixture_store("fire");
    let now = dt("2026-11-01T12:00:00Z");
    let scope = HashMap::from([("repo".to_string(), "skcapstone".to_string())]);
    let got = st
        .prefilter(
            "edit HOSTS= now",
            &scope,
            Some(now),
            &PrefilterOpts::default(),
        )
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].state, got[0].fires), (State::Fired, 1));
    let flat = st.get("int-004").unwrap().unwrap();
    assert_eq!(
        (flat.state, flat.fires, flat.last_fired_at),
        (State::Fired, 1, Some(now))
    );
    drop(st);
    let rows = table_rows(
        &base,
        "SELECT state, fires, last_fired_at FROM intentions WHERE id='int-004'",
    );
    assert_eq!(rows, vec![json!(["fired", 1, "2026-11-01T12:00:00+00:00"])]);
    // cooldown now blocks a second fire
    let st = open(&base);
    let again = st
        .prefilter(
            "HOSTS= again",
            &scope,
            Some(dt("2026-11-01T12:30:00Z")),
            &PrefilterOpts::default(),
        )
        .unwrap();
    assert!(again.is_empty());
}

#[test]
fn supersession_sweep_invalidates_int_011() {
    let (_b, st) = fixture_store("sweep");
    let changed = st
        .sweep_supersession(Some(dt("2026-11-01T12:00:00Z")))
        .unwrap();
    assert_eq!(changed, vec!["int-011".to_string()]);
    assert_eq!(
        st.get("int-011").unwrap().unwrap().state,
        State::Invalidated
    );
    // and the row follows: list reads the index
    let inv = st.list(Some("invalidated")).unwrap();
    assert_eq!(inv.len(), 1);
    assert!(st
        .sweep_supersession(Some(dt("2026-11-01T12:00:00Z")))
        .unwrap()
        .is_empty());
}

#[test]
fn terminal_states_ignore_every_event_and_save_cannot_revive() {
    let (_b, st) = fixture_store("terminal");
    for ev in [
        "arm",
        "fire",
        "snooze",
        "done",
        "cancel",
        "expire",
        "invalidate",
    ] {
        let after = st.apply("int-009", ev, None, None).unwrap().unwrap();
        assert_eq!(after.state, State::Done, "{ev}");
    }
    let mut stale = st.get("int-009").unwrap().unwrap();
    stale.state = State::Armed;
    let e = st.save(&stale).unwrap_err().to_string();
    assert!(
        e.contains("refusing state change 'done' -> 'armed' for 'int-009'"),
        "{e}"
    );
    assert!(st.apply("int-001", "bogus", None, None).is_err());
    assert!(st.apply("nope", "fire", None, None).unwrap().is_none());
    assert!(st.apply("../x", "fire", None, None).is_err());
}

#[test]
fn caps_deadline_and_empty_prompts() {
    let base = scratch("caps").join("base");
    let st = open(&base);
    for n in 0..6 {
        let mut i = Intention::new(&format!("c{n}"), Kind::Event, "t");
        i.state = State::Armed;
        i.trigger_terms = vec!["capword".into()];
        st.save(&i).unwrap();
    }
    let now = dt("2026-11-01T12:00:00Z");
    let run = |opts: PrefilterOpts, text: &str| {
        let (_b, st) = {
            let b = scratch("caps2").join("base");
            copy_dir(&base, &b);
            (b.clone(), open(&b))
        };
        st.prefilter(text, &none(), Some(now), &opts).unwrap().len()
    };
    // a generous deadline: fsync-ing fires must not be what ends these runs
    let slow = |m: Option<i64>| PrefilterOpts {
        deadline_s: Some(30.0),
        max_fires: m,
        t0: None,
    };
    assert_eq!(run(slow(None), "capword"), 3, "default cap is 3");
    assert_eq!(run(slow(Some(5)), "capword"), 5);
    assert_eq!(run(slow(Some(0)), "capword"), 0);
    assert_eq!(
        run(
            PrefilterOpts {
                deadline_s: Some(0.0),
                ..Default::default()
            },
            "capword"
        ),
        0
    );
    assert_eq!(run(PrefilterOpts::default(), ""), 0);
    assert_eq!(run(PrefilterOpts::default(), "  \n\t "), 0);
    assert_eq!(
        run(PrefilterOpts::default(), "capwords capword2 xcapword"),
        0,
        "whole tokens only"
    );
}

#[test]
fn prefilter_safe_never_fails() {
    let d = scratch("safe");
    let afile = d.join("a-file");
    fs::write(&afile, "x").unwrap();
    assert!(prefilter_safe(&afile, "anything", &none(), None).is_empty());
    assert!(
        prefilter_safe(d.join("missing/deeper"), "anything", &none(), None).is_empty()
            || d.join("missing/deeper/intentions").exists()
    );
    let (b, _st) = fixture_store("safe2");
    let got = prefilter_safe(
        &b,
        "rework the fusion weights",
        &HashMap::from([("repo".into(), "skmemory".into())]),
        Some(dt("2026-11-01T12:00:00Z")),
    );
    assert_eq!(
        got.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
        vec!["int-008"]
    );
}

// ------------------------------------------------------------------ locks

#[test]
fn lock_dir_is_the_python_path() {
    // sha1("/a/b") per Python's hashlib, first 16 hex chars.
    let d = lock_dir_for(Path::new("/a/b"), Some(Path::new("/r")));
    let uid = fs::metadata("/proc/self")
        .map(|m| std::os::unix::fs::MetadataExt::uid(&m))
        .unwrap();
    assert_eq!(
        d,
        PathBuf::from(format!("/r/skmemory-intent-locks-{uid}/a51458f8d2c1fbd8"))
    );
    // a symlink to the store resolves to the same lock dir as the store itself
    let t = scratch("linkdir");
    let real = t.join("real");
    fs::create_dir_all(&real).unwrap();
    std::os::unix::fs::symlink(&real, t.join("link")).unwrap();
    assert_eq!(
        lock_dir_for(&real, Some(Path::new("/r"))),
        lock_dir_for(&t.join("link"), Some(Path::new("/r")))
    );
    // `..` is normalised the way Path.resolve() does
    assert_eq!(
        lock_dir_for(Path::new("/a/x/../b"), Some(Path::new("/r"))),
        lock_dir_for(Path::new("/a/b"), Some(Path::new("/r")))
    );
}

#[test]
fn flock_serialises_handles_and_prefilter_try_locks() {
    let (base, st) = fixture_store("lock");
    let other = open(&base);
    assert_eq!(st.lock_dir(), other.lock_dir());
    let g = st.lock("int-004", None).unwrap();
    assert!(
        other.lock("int-004", Some(0.0)).is_err(),
        "second handle must see it busy"
    );
    assert!(
        other.lock("int-005", Some(0.0)).is_ok(),
        "locks are per intention"
    );
    let p = fs::metadata(st.lock_dir().join("int-004.lock")).unwrap();
    assert_eq!(std::os::unix::fs::MetadataExt::mode(&p) & 0o777, 0o600);
    let scope = HashMap::from([("repo".to_string(), "skcapstone".to_string())]);
    let now = dt("2026-11-01T12:00:00Z");
    // a locked intention is skipped this prompt, not fired and not waited for
    let t = std::time::Instant::now();
    let got = other
        .prefilter("HOSTS=", &scope, Some(now), &PrefilterOpts::default())
        .unwrap();
    assert!(got.is_empty());
    assert!(t.elapsed() < Duration::from_millis(500));
    assert_eq!(other.get("int-004").unwrap().unwrap().state, State::Armed);
    drop(g);
    assert_eq!(
        other
            .prefilter("HOSTS=", &scope, Some(now), &PrefilterOpts::default())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn apply_blocks_on_the_lock_then_proceeds() {
    let (base, st) = fixture_store("block");
    let g = st.lock("int-005", None).unwrap();
    let b2 = base.clone();
    let h = std::thread::spawn(move || {
        let s2 = open(&b2);
        s2.apply("int-005", "done", None, None)
            .unwrap()
            .unwrap()
            .state
    });
    std::thread::sleep(Duration::from_millis(150));
    assert!(!h.is_finished(), "apply must wait for the lock");
    drop(g);
    assert_eq!(h.join().unwrap(), State::Done);
}

#[test]
fn tick_lock_is_host_local_and_shared() {
    let (_b, st) = fixture_store("ticklock");
    let g = st.lock_named("_tick.lock", Some(0.0)).unwrap();
    let mut ok = |_: &str| true;
    let r = run_tick(
        &st,
        TickOptions {
            deliver: Some(&mut ok),
            any_host: true,
            now: Some(dt("2026-11-01T12:00:00Z")),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        r.skipped.as_deref(),
        Some("another tick is running on this host")
    );
    assert!(r.fired.is_empty());
    drop(g);
    let mut ok = |_: &str| true;
    let r = run_tick(
        &st,
        TickOptions {
            deliver: Some(&mut ok),
            any_host: true,
            now: Some(dt("2026-11-01T12:00:00Z")),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.fired, vec!["int-002".to_string()]);
}

// ------------------------------------------------------------------- tick

#[test]
fn tick_delivers_before_it_records() {
    let (_b, st) = fixture_store("order");
    let now = dt("2026-11-01T12:00:00Z");
    let mut seen_state = None;
    let mut deliver = |_: &str| {
        // at delivery time the intention is still due: nothing recorded yet
        seen_state = Some(st.get("int-002").unwrap().unwrap().state);
        true
    };
    let r = run_tick(
        &st,
        TickOptions {
            now: Some(now),
            deliver: Some(&mut deliver),
            any_host: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.fired, vec!["int-002".to_string()]);
    assert_eq!(seen_state, Some(State::Pending));
    assert_eq!(st.get("int-002").unwrap().unwrap().state, State::Fired);
}

#[test]
fn tick_failure_changes_nothing_and_stops() {
    let (_b, st) = fixture_store("fail");
    let mut calls = 0;
    let mut deliver = |_: &str| {
        calls += 1;
        false
    };
    let r = run_tick(
        &st,
        TickOptions {
            now: Some(dt("2026-11-01T12:00:00Z")),
            deliver: Some(&mut deliver),
            any_host: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        (calls, r.fired.len(), r.failed.clone()),
        (1, 0, vec!["int-002".to_string()])
    );
    assert_eq!(st.get("int-002").unwrap().unwrap().state, State::Pending);
}

#[test]
fn tick_repeating_intention_rearms_skipping_missed_intervals() {
    let base = scratch("repeat").join("base");
    let st = open(&base);
    let mut i = Intention::new("r1", Kind::Time, "daily");
    i.state = State::Armed;
    i.due_at = Some(dt("2026-10-01T12:00:00Z"));
    i.interval_s = Some(86_400);
    st.save(&i).unwrap();
    let now = dt("2026-11-01T12:30:00Z");
    let mut sent = 0;
    let mut deliver = |_: &str| {
        sent += 1;
        true
    };
    let r = run_tick(
        &st,
        TickOptions {
            now: Some(now),
            deliver: Some(&mut deliver),
            any_host: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        (r.fired.len(), sent),
        (1, 1),
        "one fire for a month of missed days, not 31"
    );
    let after = st.get("r1").unwrap().unwrap();
    assert_eq!(after.state, State::Armed);
    assert_eq!(after.due_at, Some(dt("2026-11-02T12:00:00Z")));
    assert_eq!(after.fires, 1);
}

#[test]
fn sk_alert_runs_an_argv_without_a_shell() {
    use std::os::unix::fs::PermissionsExt;
    let d = scratch("alert");
    let log = d.join("argv.txt");
    let script = d.join("fake-alert");
    fs::write(&script, format!("#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\" >> {}; done\nexit ${{FAKE_EXIT:-0}}\n", log.display())).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    // one test owns SKMEMORY_ALERT_CMD: nothing else in this binary reads it
    std::env::set_var(
        "SKMEMORY_ALERT_CMD",
        format!("{} 'two words' --flag", script.display()),
    );
    let evil = format!(
        "x $(touch {}/pwned); `touch {}/pwned2` ; rm -rf /nonexistent",
        d.display(),
        d.display()
    );
    assert!(sk_alert_deliver(&evil, Duration::from_secs(5)));
    let argv: Vec<String> = fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(
        argv,
        vec![
            "two words",
            "--flag",
            "-g",
            "intentions",
            "--title",
            "skmemory intention due",
            &evil
        ]
    );
    assert!(
        !d.join("pwned").exists() && !d.join("pwned2").exists(),
        "no shell may run"
    );
    // non-zero exit, missing command, timeout: all false
    std::env::set_var("SKMEMORY_ALERT_CMD", "/bin/false");
    assert!(!sk_alert_deliver("t", Duration::from_secs(5)));
    std::env::set_var("SKMEMORY_ALERT_CMD", "/nonexistent/sk-alert-xyz");
    assert!(!sk_alert_deliver("t", Duration::from_secs(5)));
    std::env::set_var("SKMEMORY_ALERT_CMD", "/bin/sleep 5");
    let t = std::time::Instant::now();
    assert!(!sk_alert_deliver("t", Duration::from_millis(300)));
    assert!(
        t.elapsed() < Duration::from_secs(3),
        "timeout must kill the child"
    );
    std::env::set_var("SKMEMORY_ALERT_CMD", "/bin/true");
    assert!(sk_alert_deliver("t", Duration::from_secs(5)));
    std::env::remove_var("SKMEMORY_ALERT_CMD");
}

// ----------------------------------------------------------------- ledger

#[test]
fn outcome_fixture_scores() {
    let exp: Value = serde_json::from_str(
        &fs::read_to_string(fixtures().join("outcomes-expected-v1.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(exp["half_life_days"], 90);
    assert_eq!(ledger::HALF_LIFE_DAYS, 90.0);
    assert_eq!(exp["harm_weight"], 4.0);
    assert_eq!(ledger::HARM_WEIGHT, 4.0);
    let base = scratch("outcomes").join("base");
    fs::create_dir_all(base.join("ledger")).unwrap();
    fs::copy(
        fixtures().join("outcomes-v1.jsonl"),
        base.join("ledger/outcomes-skap01-2026-08.jsonl"),
    )
    .unwrap();
    let l = Ledger::new(&base, Some("skap01"), &own());
    assert_eq!(
        l.events().len(),
        34,
        "id-less lines get stable hash ids and none collide"
    );
    let scores = l.scores(Some(dt(exp["as_of"].as_str().unwrap())));
    assert_eq!(scores.len(), 4);
    for (key, want) in exp["expected"].as_object().unwrap() {
        let s = scores
            .iter()
            .find(|s| &s.key() == key)
            .unwrap_or_else(|| panic!("{key}"));
        assert_eq!(
            s.follow_rate,
            want["follow_rate"].as_f64().unwrap(),
            "{key}"
        );
        assert_eq!(
            s.net_effect_sign,
            want["net_effect_sign"].as_str().unwrap(),
            "{key}"
        );
        assert_eq!(
            s.anti_pattern,
            want["anti_pattern"].as_bool().unwrap(),
            "{key}"
        );
        assert_eq!(s.rank as u64, want["rank"].as_u64().unwrap(), "{key}");
        assert_eq!(s.flag, want["flag"].as_str(), "{key}");
    }
}

#[test]
fn ledger_rules() {
    let base = scratch("ledger-rules").join("base");
    let l = Ledger::new(&base, Some("skap01"), &own());
    fn r(effect: Option<&'static str>, s: &str) -> Record<'static> {
        Record {
            effect,
            session: "s",
            at: Some(dt(s)),
            ..Default::default()
        }
    }
    // validation
    assert!(l
        .record("a", "1", "bogus", &r(None, "2026-10-01T00:00:00Z"))
        .is_err());
    assert!(l
        .record(
            "a",
            "1",
            "followed",
            &r(Some("bogus"), "2026-10-01T00:00:00Z")
        )
        .is_err());
    assert!(l
        .record(
            "a",
            "1",
            "served",
            &r(Some("success"), "2026-10-01T00:00:00Z")
        )
        .is_err());
    assert!(l
        .record("", "1", "served", &r(None, "2026-10-01T00:00:00Z"))
        .is_err());
    assert!(l
        .record("a", "", "served", &r(None, "2026-10-01T00:00:00Z"))
        .is_err());
    // per host, per month
    l.record("a", "1", "served", &r(None, "2026-10-31T23:59:59Z"))
        .unwrap();
    l.record("a", "1", "served", &r(None, "2026-11-01T00:00:00Z"))
        .unwrap();
    let mut names: Vec<String> = fs::read_dir(base.join("ledger"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "outcomes-skap01-2026-10.jsonl",
            "outcomes-skap01-2026-11.jsonl"
        ]
    );
    // a host name is sanitised into the file name
    let odd = Ledger::new(&base, Some("sk/ap 01"), &own());
    assert_eq!(odd.host(), "sk-ap-01");
    // floors: 2 harmful follows are not an anti-pattern, 3 are
    for n in 0..2 {
        l.record(
            "h",
            "1",
            "followed",
            &r(Some("harm"), &format!("2026-10-0{}T00:00:00Z", n + 1)),
        )
        .unwrap();
    }
    let s = l.score("h", "1", Some(dt("2026-11-01T00:00:00Z"))).unwrap();
    assert!(!s.anti_pattern && s.flag.is_none());
    l.record(
        "h",
        "1",
        "followed",
        &r(Some("harm"), "2026-10-03T00:00:00Z"),
    )
    .unwrap();
    assert!(
        l.score("h", "1", Some(dt("2026-11-01T00:00:00Z")))
            .unwrap()
            .anti_pattern
    );
    // followed_error counts as a follow but is neutral: no harm, no success
    l.record("e", "1", "served", &r(None, "2026-10-01T00:00:00Z"))
        .unwrap();
    for n in 0..3 {
        l.record(
            "e",
            "1",
            "followed_error",
            &r(None, &format!("2026-10-0{}T00:00:00Z", n + 1)),
        )
        .unwrap();
    }
    let s = l.score("e", "1", Some(dt("2026-11-01T00:00:00Z"))).unwrap();
    assert_eq!(
        (s.followed, s.net, s.anti_pattern, s.net_effect_sign),
        (3, 0.0, false, "0")
    );
    // served-never-followed needs 3 serves
    for n in 0..2 {
        l.record(
            "n",
            "1",
            "served",
            &r(None, &format!("2026-10-0{}T00:00:00Z", n + 1)),
        )
        .unwrap();
    }
    assert!(l
        .score("n", "1", Some(dt("2026-11-01T00:00:00Z")))
        .unwrap()
        .flag
        .is_none());
    l.record("n", "1", "served", &r(None, "2026-10-04T00:00:00Z"))
        .unwrap();
    assert_eq!(
        l.score("n", "1", Some(dt("2026-11-01T00:00:00Z")))
            .unwrap()
            .flag,
        Some("served-never-followed")
    );
    // a duplicated id is one event; two genuine identical events stay two
    let id = l
        .record(
            "d",
            "1",
            "served",
            &Record {
                event_id: Some("dup1"),
                ..r(None, "2026-10-01T00:00:00Z")
            },
        )
        .unwrap();
    l.record(
        "d",
        "1",
        "served",
        &Record {
            event_id: Some(&id),
            ..r(None, "2026-10-01T00:00:00Z")
        },
    )
    .unwrap();
    assert_eq!(l.events().iter().filter(|e| e.skill == "d").count(), 1);
    l.record("d2", "1", "served", &r(None, "2026-10-01T00:00:00Z"))
        .unwrap();
    l.record("d2", "1", "served", &r(None, "2026-10-01T00:00:00Z"))
        .unwrap();
    assert_eq!(l.events().iter().filter(|e| e.skill == "d2").count(), 2);
    // harm is only ever explicit: nothing here wrote one for "e"
    assert!(l
        .events()
        .iter()
        .filter(|e| e.skill == "e")
        .all(|e| e.effect.is_none()));
}

fn own() -> skmem::audience::OwnerScope {
    skmem::audience::OwnerScope::resolve(&skmem::env::Env::empty()).unwrap()
}
