//! The scoring-relevant half of `skmemory/ledger_sidecar.py` (SKMEM-R-07):
//! event ids, source names, and the idempotent Hermes usage import.
//!
//! What the signals mean (read before trusting a score): a transcript
//! `followed` is a HEURISTIC and is stored with `effect=null`; `success` is
//! never written by a hook and `harm` only by an explicit record from an agent
//! or user. A matched command that came back with an error is
//! `followed_error`, also effect null, so a grep miss cannot brand a good
//! skill harmful. Counters imported from Hermes are scored apart (see
//! [`super::IMPORTED_SOURCES`]). The transcript scanner is a Claude Code hook
//! concern and stays in Python.

use super::{hex32, Ledger, LedgerError, Record};
use crate::pytime::Dt;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub const SRC_SKILL: &str = "claude-stop:skill-tool";
pub const SRC_HEURISTIC: &str = "heuristic:bash-command-match";
pub const SRC_ERROR: &str = "tool-error:matched-command";
pub const SRC_HERMES: &str = "hermes-usage-counter";

/// `_sha`: first 32 hex chars of the SHA-256 of the parts joined by U+001F.
pub fn sha(parts: &[&str]) -> String {
    hex32(parts.join("\x1f").as_bytes())
}

/// Counts of one [`import_hermes_usage`] run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImportStats {
    pub skills: i64,
    pub served: i64,
    pub followed: i64,
    pub skipped: i64,
}

fn int_of(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

fn text_of(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Import `.usage.json` as served/followed events. Opens the file read-only.
///
/// Real shape: `{skill: {use_count, view_count, last_used_at, last_viewed_at,
/// patch_count, ...}}`. There are no per-use timestamps, only the last one, so
/// each counter becomes N events keyed `(skill, kind, ordinal)`: re-importing
/// adds only ordinals beyond what is already held. `view_count` is served,
/// `use_count` is followed; where use exceeds view, served is raised to use.
/// The version is `patch-<patch_count>`.
pub fn import_hermes_usage(usage_path: &Path, ledger: &Ledger) -> Result<ImportStats, LedgerError> {
    let data: Value = serde_json::from_slice(&fs::read(usage_path)?)
        .map_err(|e| LedgerError::Value(e.to_string()))?;
    let obj = data
        .as_object()
        .ok_or_else(|| LedgerError::Value("usage file is not an object".into()))?;
    let mut held: HashSet<String> = ledger.events().into_iter().map(|e| e.id).collect();
    let mut stats = ImportStats::default();
    let mut names: Vec<&String> = obj.keys().collect();
    names.sort();
    for name in names {
        let Some(st) = obj[name].as_object() else {
            stats.skipped += 1;
            continue;
        };
        let get = |k: &str| st.get(k);
        let used = int_of(get("use_count"));
        let served = int_of(get("view_count")).max(used);
        let at = text_of(get("last_used_at"))
            .or_else(|| text_of(get("last_viewed_at")))
            .or_else(|| text_of(get("created_at")));
        let Some(at) = at.filter(|_| served != 0) else {
            stats.skipped += 1;
            continue;
        };
        stats.skills += 1;
        let version = format!("patch-{}", int_of(get("patch_count")));
        for (kind, n) in [("served", served), ("followed", used)] {
            let when = if kind == "served" {
                text_of(get("last_viewed_at")).unwrap_or(at)
            } else {
                at
            };
            let when = Dt::parse(when)
                .ok_or_else(|| LedgerError::Value(format!("invalid isoformat string: {when:?}")))?;
            for i in 0..n {
                let eid = sha(&["hermes", name, kind, &i.to_string()]);
                if held.contains(&eid) {
                    continue;
                }
                ledger.record(
                    name,
                    &version,
                    kind,
                    &Record {
                        session: "hermes-usage",
                        at: Some(when),
                        event_id: Some(&eid),
                        source: Some(SRC_HERMES),
                        missing: Some(
                            "per-use time and outcome unknown; at is the last recorded use",
                        ),
                        ..Default::default()
                    },
                )?;
                held.insert(eid);
                if kind == "served" {
                    stats.served += 1;
                } else {
                    stats.followed += 1;
                }
            }
        }
    }
    Ok(stats)
}
