//! Source-tier occupancy cap: the port of `skmemory/tiers.py` (`apply_tier_cap`).
//!
//! Web and self-generated memories may take at most 40 percent of the returned
//! slots WHEN higher-tier hits exist to fill the rest. It is an occupancy limit,
//! never a filter: with too few higher-tier hits the capped ones backfill in
//! rank order. A schema v1 row carries a placeholder tier and is uncapped unless
//! its `source` names a known generator.

use crate::env::Env;
use crate::search::Candidate;
use std::collections::HashMap;

/// Fraction of the page capped tiers may occupy.
pub const CAP_FRACTION: f64 = 0.4;

const GENERATED_SOURCES: [&str; 8] = [
    "skwhisper",
    "journal-synthesis",
    "dreaming-engine",
    "consolidation",
    "claude-code-hook",
    "mcp:memory_save_session",
    "mcp:memory_pre_compact",
    "observation",
];

/// The tier retrieval should honour, or `None` when the row is uncapped (legacy).
pub fn effective_tier(c: &Candidate) -> Option<&str> {
    let sv = if c.schema_version == 0 { 1 } else { c.schema_version };
    if sv >= 2 {
        return c.source_tier.as_deref();
    }
    let src = c.source.trim().to_lowercase();
    if GENERATED_SOURCES.contains(&src.as_str()) {
        return Some("self-generated");
    }
    None
}

/// Whether the row's effective tier is `web` or `self-generated`.
pub fn is_capped(c: &Candidate) -> bool {
    matches!(effective_tier(c), Some("web" | "self-generated"))
}

/// Pick the top `limit` ids, capping web/self-generated occupancy. Relative
/// rank is preserved; ids not in `found` count as uncapped. `SKMEMORY_TIER_CAP`
/// in `0`, `off`, `false`, `no` turns the cap off (plain truncation).
pub fn apply_tier_cap(
    ids: &[String],
    found: &HashMap<String, Candidate>,
    limit: usize,
    env: &Env,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let flag = env.get_or("SKMEMORY_TIER_CAP", "on").trim().to_lowercase();
    if matches!(flag.as_str(), "0" | "off" | "false" | "no") {
        return ids.iter().take(limit).cloned().collect();
    }
    let cap = (CAP_FRACTION * limit as f64 + 1e-9).floor() as usize;
    let mut chosen: Vec<usize> = Vec::new();
    let mut skipped: Vec<usize> = Vec::new();
    let mut n_capped = 0;
    for (i, id) in ids.iter().enumerate() {
        if chosen.len() >= limit {
            break;
        }
        if found.get(id).is_some_and(is_capped) {
            if n_capped < cap {
                n_capped += 1;
                chosen.push(i);
            } else {
                skipped.push(i);
            }
        } else {
            chosen.push(i);
        }
    }
    for i in skipped {
        if chosen.len() >= limit {
            break;
        }
        chosen.push(i);
    }
    chosen.sort_unstable();
    chosen.into_iter().map(|i| ids[i].clone()).collect()
}
