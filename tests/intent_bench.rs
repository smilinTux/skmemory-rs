//! SKMEM-R-07: prefilter timing. Ignored by default (it measures, it does not
//! assert a number): run with
//! `cargo test --release --test intent_bench -- --ignored --nocapture`.
//! With `SKMEM_PYTHON` (and `SKMEM_PYTHONPATH`) set it times the Python
//! prefilter on the same store for comparison.

mod intent_common;

use intent_common::*;
use skmem::intentions::{prefilter_safe, Intention, IntentionStore, Kind, PrefilterOpts, State};
use std::collections::HashMap;
use std::time::Instant;

const N: usize = 2000;

fn med(runs: usize, mut f: impl FnMut()) -> f64 {
    let mut xs: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (xs[xs.len() / 2] * 1000.0).round() / 1000.0
}

#[test]
#[ignore = "timing run, not a gate"]
fn prefilter_timing() {
    let base = scratch("bench").join("base");
    let st =
        IntentionStore::open_with(&base, 30_000, Some(&base.join("../locks")), &own()).unwrap();
    for n in 0..N {
        let mut i = Intention::new(&format!("b{n:05}"), Kind::Event, "bench");
        i.state = State::Armed;
        i.trigger_terms = vec![
            format!("alpha{} beta", n % 50),
            format!("zq{n} unique"),
            format!("delta{n} epsilon"),
        ];
        i.scope = vec![("repo".into(), "x".into())];
        st.save(&i).unwrap();
    }
    st.reindex().unwrap();
    drop(st);
    let short = "please look at the rollout notes and also alpha7 gamma then check delta7 and epsilon things ".repeat(3);
    let hit_scope_fail = "talk about alpha9 beta today".to_string();
    let long = format!(
        "{} end",
        "lorem ipsum dolor sit amet alpha3 gamma ".repeat(3000)
    );
    let runs = 25;
    let none = HashMap::new();
    let mut rows = Vec::new();
    for (name, text) in [
        ("short_nomatch", &short),
        ("phrase_hit_scope_fail", &hit_scope_fail),
        ("long_100kb", &long),
    ] {
        let cold = med(runs, || {
            let _ = prefilter_safe(&base, text, &none, None);
        });
        let st =
            IntentionStore::open_with(&base, 30_000, Some(&base.join("../locks")), &own()).unwrap();
        let opts = PrefilterOpts {
            deadline_s: Some(5.0),
            ..Default::default()
        };
        st.prefilter(text, &none, None, &opts).unwrap();
        let warm = med(runs, || {
            st.prefilter(text, &none, None, &opts).unwrap();
        });
        rows.push((name, cold, warm));
    }
    println!("rust prefilter, {N} armed event intentions, median of {runs} (ms):");
    for (n, c, w) in &rows {
        println!("  {n:24} cold {c:8.3}   warm {w:8.3}");
    }
    if let Ok(py) = std::env::var("SKMEM_PYTHON") {
        let out = std::process::Command::new(py)
            .env(
                "PYTHONPATH",
                std::env::var("SKMEM_PYTHONPATH").unwrap_or_default(),
            )
            .arg(parity().join("bench_prefilter.py"))
            .arg(&base)
            .arg(runs.to_string())
            .output()
            .unwrap();
        println!(
            "python prefilter, same store (ms): {}",
            String::from_utf8_lossy(&out.stdout).trim()
        );
        if !out.status.success() {
            println!("{}", String::from_utf8_lossy(&out.stderr));
        }
    }
}

fn own() -> skmem::audience::OwnerScope {
    skmem::audience::OwnerScope::resolve(&skmem::env::Env::empty()).unwrap()
}
