//! SKMEM-R-04: `as_of` parsing against Python's `normalize_as_of` (hermetic).
//! `tests/pg_parity/asof_norm.json` is produced by `gen_pg_parity.py`.

use serde_json::Value;
use skmem::validity::normalize_as_of;
use std::path::Path;

/// Python accepts these degenerate shapes; Rust refuses them (stricter, never
/// looser than Python), so they are the only allowed disagreement.
const STRICTER: [&str; 2] = ["2026-04-10T05:00:00 +05:00", "2026-04-10Z"];

#[test]
fn normalize_as_of_matches_python() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pg_parity/asof_norm.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let cases = cases.as_array().unwrap();
    assert!(cases.len() > 40);
    for c in cases {
        let input = c["input"].as_str().unwrap();
        let got = normalize_as_of(input);
        if c.get("error").is_some() {
            assert!(got.is_err(), "{input:?} must be rejected, got {got:?}");
        } else if STRICTER.contains(&input) {
            assert!(got.is_err(), "{input:?}");
        } else {
            let want = c["out"].as_str().map(|s| s.to_string());
            assert_eq!(
                got.unwrap_or_else(|e| panic!("{input:?}: {e}")),
                want,
                "{input:?}"
            );
        }
    }
}
