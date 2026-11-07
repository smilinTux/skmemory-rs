//! Decay channels: the port of `skmemory/decay.py` (`decay_score`, `rerank`,
//! `configured_channel`).
//!
//! `SKMEMORY_DECAY` selects a channel (unset or `0/off/false/no/disabled` is off).
//! The decay rerank bends a relevance-ordered list: each item gets a positional
//! prior `1 / (1 + rank)` and the final key is `prior * score ** gamma`, so only
//! near ties move. It never adds or drops a hit.

use crate::env::Env;
use crate::pytime::Dt;
use crate::search::Candidate;
use std::collections::HashMap;

pub const FACT_HALF_LIFE_DAYS: f64 = 30.0;
pub const GIST_HALF_LIFE_DAYS: f64 = 120.0;
pub const AFFECT_FLOOR: f64 = 0.15;
pub const AFFECT_GATE: f64 = 0.5;
/// How strongly decay bends a relevance-ordered list.
pub const RERANK_GAMMA: f64 = 0.15;

/// A decay channel. `None_` is the "none" channel (score 1 for everything).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    None_,
    Recency,
    Reinforcement,
    AffectWeighted,
}

impl Channel {
    pub fn parse(s: &str) -> Option<Channel> {
        match s {
            "none" => Some(Channel::None_),
            "recency" => Some(Channel::Recency),
            "reinforcement" => Some(Channel::Reinforcement),
            "affect_weighted" => Some(Channel::AffectWeighted),
            _ => None,
        }
    }
}

/// Usage counters for one memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub access_count: i64,
    pub last_accessed: Option<String>,
}

/// `configured_channel`: the channel `SKMEMORY_DECAY` names, `None` when off.
pub fn configured_channel(env: &Env) -> Option<Channel> {
    let raw = env.get("SKMEMORY_DECAY")?;
    let val = raw.trim().to_lowercase();
    match val.as_str() {
        "0" | "off" | "false" | "no" | "disabled" => None,
        "" | "1" | "on" | "true" | "yes" => Some(Channel::AffectWeighted),
        other => Channel::parse(other).filter(|c| *c != Channel::None_),
    }
}

fn days(as_of: Dt, ts: Option<Dt>) -> f64 {
    match ts {
        None => 0.0,
        Some(t) => ((as_of.micros - t.micros) as f64 / 1e6 / 86400.0).max(0.0),
    }
}

/// `decay_score`. `intensity` is on 0-1; `gist` selects the long half-life.
pub fn decay_score(
    channel: Channel,
    as_of: Dt,
    created_at: Option<&str>,
    last_accessed: Option<&str>,
    access_count: i64,
    intensity: f64,
    gist: bool,
) -> f64 {
    if channel == Channel::None_ {
        return 1.0;
    }
    let created = created_at.and_then(Dt::parse);
    let half = if gist {
        GIST_HALF_LIFE_DAYS
    } else {
        FACT_HALF_LIFE_DAYS
    };
    if channel == Channel::Recency {
        return 0.5f64.powf(days(as_of, created) / half);
    }
    let accessed = last_accessed.and_then(Dt::parse);
    let touch = [created, accessed]
        .into_iter()
        .flatten()
        .max_by_key(|d| d.micros);
    let f = 0.5f64.powf(days(as_of, touch) / half);
    let boost = 1.0 + (access_count.max(0) as f64).ln_1p();
    if channel == Channel::Reinforcement {
        return f * boost;
    }
    let i = intensity.clamp(0.0, 1.0);
    let floor = if i >= AFFECT_GATE { AFFECT_FLOOR } else { 0.0 };
    f.max(floor) * boost * (1.0 + i)
}

/// `rerank`: bend a relevance-ordered id list by decay. `usage` of `None` (the
/// primary keeps no counters) or fewer than two items leaves the order alone.
pub fn rerank(
    ids: &[String],
    found: &HashMap<String, Candidate>,
    usage: Option<&HashMap<String, Usage>>,
    channel: Channel,
    as_of: Dt,
    gamma: f64,
) -> Vec<String> {
    let Some(usage) = usage else {
        return ids.to_vec();
    };
    if channel == Channel::None_ || ids.len() < 2 {
        return ids.to_vec();
    }
    let default = Usage::default();
    let mut keyed: Vec<(f64, usize)> = ids
        .iter()
        .enumerate()
        .map(|(rank, id)| {
            let u = usage.get(id).unwrap_or(&default);
            let c = found.get(id);
            let s = decay_score(
                channel,
                as_of,
                c.and_then(|c| c.created_at.as_deref()),
                u.last_accessed.as_deref(),
                u.access_count,
                c.map_or(0.0, |c| c.intensity) / 10.0,
                c.is_some_and(|c| c.layer == "long-term"),
            );
            (-(1.0 / (1 + rank) as f64) * s.max(1e-9).powf(gamma), rank)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, r)| ids[r].clone()).collect()
}
