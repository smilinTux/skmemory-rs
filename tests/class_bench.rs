//! SKMEM-R-06B: the per-class bench over the vendored synthetic v1 fixtures.
//!
//! `tests/fixtures/eval-v1/class-bench-python.json` is written by
//! `tests/golden/gen_class_bench_golden.py` from the current Python skmemory
//! (`skmemory.eval.class_bench`). The gate: for every class and every metric
//! the Python candidate reports, the Rust candidate must meet or beat it. The
//! page-for-page test is the stronger check that the pipeline is the same.
//! Hermetic: no Python, no network, no Postgres.

use serde_json::Value;
use skmem::bench::{self, BenchResult};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const EPS: f64 = 1e-9;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval-v1")
}

fn golden() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("class-bench-python.json")).unwrap())
        .unwrap()
}

fn result() -> &'static BenchResult {
    static R: OnceLock<BenchResult> = OnceLock::new();
    R.get_or_init(|| bench::run_bench(&dir()).expect("run_bench"))
}

#[test]
fn hashed_embedding_buckets_match_python_blake2b() {
    for p in golden()["bucket_probes"].as_array().unwrap() {
        let concept = p["concept"].as_str().unwrap();
        assert_eq!(
            bench::concept_bucket(concept) as u64,
            p["bucket"].as_u64().unwrap(),
            "{concept}"
        );
    }
}

#[test]
fn time_windows_match_python() {
    let now = {
        // 2026-11-01T12:00:00Z
        let (y, m, d) = (2026i64, 11i64, 1i64);
        let days = {
            let yy = if m <= 2 { y - 1 } else { y };
            let era = yy.div_euclid(400);
            let yoe = yy - era * 400;
            let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
            era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
        };
        days * 86_400 + 12 * 3600
    };
    let g = golden();
    let queries = std::fs::read_to_string(dir().join("queries-v1.jsonl")).unwrap();
    let mut some = 0;
    for line in queries.lines().filter(|l| !l.trim().is_empty()) {
        let q: Value = serde_json::from_str(line).unwrap();
        let id = q["id"].as_str().unwrap();
        let want = &g["time_windows"][id];
        let got = bench::time_window(q["query"].as_str().unwrap(), now);
        match (want.as_array(), got) {
            (None, None) => {}
            (Some(w), Some((a, b))) => {
                some += 1;
                assert_eq!(
                    (w[0].as_str().unwrap(), w[1].as_str().unwrap()),
                    (a.as_str(), b.as_str()),
                    "{id}"
                );
            }
            (w, g) => panic!("{id}: python {w:?} rust {g:?}"),
        }
    }
    assert!(some >= 5, "only {some} windows");
}

#[test]
fn candidate_meets_or_beats_python_per_class() {
    let res = result();
    let g = golden();
    let mut worse = Vec::new();
    for (class, entry) in g["by_class"].as_object().unwrap() {
        let py = &entry["strategies"]["candidate"];
        let (n, rust) = res
            .by_class
            .get(class)
            .unwrap_or_else(|| panic!("class {class}"));
        assert_eq!(
            *n as u64,
            entry["queries"].as_u64().unwrap(),
            "{class}: query count"
        );
        let rs = &rust["candidate"];
        for (metric, got) in [
            ("recall@5", rs.recall_at_5),
            ("mrr", rs.mrr),
            ("abstain_correct", rs.abstain_correct),
        ] {
            let Some(want) = py[metric].as_f64() else {
                assert!(
                    got.is_none(),
                    "{class} {metric}: python has none, rust {got:?}"
                );
                continue;
            };
            let got = got.unwrap_or_else(|| panic!("{class} {metric}: rust has none"));
            if got < want - EPS {
                worse.push(format!(
                    "{class} {metric}: rust {got:.3} < python {want:.3}"
                ));
            }
        }
    }
    assert!(worse.is_empty(), "Rust below Python: {worse:#?}");
}

#[test]
fn privacy_leaks_nothing_to_the_public_reader() {
    assert_eq!(result().leak_count, 0);
    assert_eq!(golden()["leak_count"].as_u64(), Some(0));
}

#[test]
fn candidate_pages_match_python_page_for_page() {
    let res = result();
    let g = golden();
    let mut diff = Vec::new();
    for row in &res.rows {
        let want: Vec<&str> = g["candidate_pages"][&row.id]
            .as_array()
            .unwrap_or_else(|| panic!("{}", row.id))
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let got: Vec<&str> = row.candidate_page.iter().map(String::as_str).collect();
        if got != want {
            diff.push(format!("{}: rust {got:?} python {want:?}", row.id));
        }
    }
    assert!(
        diff.is_empty(),
        "{} pages differ:\n{}",
        diff.len(),
        diff.join("\n")
    );
}

#[test]
fn table_prints_every_class() {
    let t = bench::format_table(result());
    for c in [
        "absent",
        "affect",
        "conflict",
        "entity",
        "lexical",
        "multihop",
        "privacy",
        "procedural",
        "semantic",
        "temporal",
    ] {
        assert!(t.contains(&format!("\n{c} ")), "{c} missing:\n{t}");
    }
}
