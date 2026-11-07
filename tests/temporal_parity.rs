//! SKMEM-R-07C: the Rust temporal window against Python `parse_time_window(strict=True)`.
//!
//! `tests/golden/temporal_windows.json` is written by `gen_temporal_windows.py` (thousands of
//! queries over every phrase family). The contract: Rust returns EXACTLY Python's window or
//! abstains. A different window is a bug; an abstain where Python answers is allowed (and counted).

use serde_json::Value;
use skmem::bench::time_window;
use std::path::Path;

// 2026-11-01T12:00:00Z
const NOW: i64 = 1_793_534_400;

#[test]
fn never_a_wrong_window() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/temporal_windows.json");
    let rows: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let (mut exact, mut abstain, mut py_none, mut wrong) = (0, 0, 0, Vec::new());
    for r in &rows {
        let q = r["q"].as_str().unwrap();
        let got = time_window(q, NOW);
        match (&r["w"], got) {
            (Value::Array(w), Some((a, b))) => {
                if w[0].as_str() == Some(&a) && w[1].as_str() == Some(&b) {
                    exact += 1;
                } else {
                    wrong.push(format!("{q:?}: python {w:?} rust ({a}, {b})"));
                }
            }
            (Value::Array(_), None) => abstain += 1,
            (_, None) => py_none += 1,
            (_, Some(g)) => wrong.push(format!("{q:?}: python none, rust {g:?}")),
        }
    }
    eprintln!(
        "temporal parity: {} queries, {exact} exact, {abstain} abstained where Python answers, \
         {py_none} both none, {} wrong",
        rows.len(),
        wrong.len()
    );
    assert!(
        wrong.is_empty(),
        "{} wrong windows:\n{}",
        wrong.len(),
        wrong
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        exact * 4 > rows.len(),
        "answers too rarely: only {exact} exact of {}",
        rows.len()
    );
}
