//! The post-arm half of `MemoryStore.search`: fuse, rerank, demote superseded,
//! decay, demote again, tier cap, procedure order. The same order Python applies
//! them in (see `skmemory/eval/arm_diff.py` for the replay).
//!
//! The caller runs the arms (SQLite, Postgres, graph) and hydrates the
//! candidates; [`search`] only decides the final order and says how it got there.
//! The reader ceiling and the query sanitizer are NOT optional here (SKMEM-R-06C):
//! [`SearchInput::scope`] is required, [`search`] drops every candidate and arm
//! id the reader may not see before fusing (a hidden id never influences a
//! visible rank), and the query is sanitized once on the way in, as Python's
//! `_search_unscoped` does. Out of scope: access reinforcement and the legacy
//! fallback chain that Python takes when no arm contributed (this returns
//! `None` for that case).

use crate::audience::ReaderScope;
use crate::decay::{self, Usage};
use crate::env::Env;
use crate::fusion::{self, ArmOutcome, Route, DEFAULT_ARM_CAP, RRF_K};
use crate::model::Memory;
use crate::procedures::order_procedures;
use crate::pytime::Dt;
use crate::query_sanitizer::sanitize_query;
use crate::rerank::Reranker;
use crate::tiers::apply_tier_cap;
use std::collections::{HashMap, HashSet};

/// The fields of a hydrated memory the retrieval stages read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub title: String,
    pub content: String,
    pub created_at: Option<String>,
    /// `short-term`, `mid-term` or `long-term`.
    pub layer: String,
    /// Emotional intensity on the 0-10 scale.
    pub intensity: f64,
    pub kind: String,
    pub outcome: Option<String>,
    pub valid_to: Option<String>,
    pub supersedes_id: Option<String>,
    /// `metadata.superseded_by.id`.
    pub superseded_by_id: Option<String>,
    pub source: String,
    pub source_tier: Option<String>,
    pub schema_version: i64,
    /// The audience tag. Empty (the default) is visible to the owner only.
    pub context_tag: String,
    /// Free-form tags, checked against a reader's never-share list.
    pub tags: Vec<String>,
}

impl Candidate {
    /// Project a stored [`Memory`].
    pub fn from_memory(m: &Memory) -> Candidate {
        let sup_by = m
            .metadata
            .get("superseded_by")
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        Candidate {
            id: m.id.clone(),
            title: m.title.clone(),
            content: m.content.clone(),
            created_at: Some(m.created_at.clone()),
            layer: m.layer.as_str().to_string(),
            intensity: m.emotional.intensity,
            kind: m.kind.clone(),
            outcome: m.outcome.clone(),
            valid_to: m.valid_to.clone(),
            supersedes_id: m.supersedes_id.clone(),
            superseded_by_id: sup_by,
            source: m.source.clone(),
            source_tier: Some(m.source_tier.clone()),
            schema_version: m.schema_version,
            context_tag: m.context_tag.clone(),
            tags: m.tags.clone(),
        }
    }
}

/// `_demote_superseded`: rank a closed record directly below its open head,
/// never above it. Only reorders; a record whose head is not in the pool keeps
/// its place. Stable.
pub fn demote_superseded(ids: &[String], found: &HashMap<String, Candidate>) -> Vec<String> {
    let mut succ: HashMap<String, String> = HashMap::new();
    for i in ids {
        if let Some(prev) = found
            .get(i)
            .and_then(|m| m.supersedes_id.as_deref())
            .filter(|s| !s.is_empty())
        {
            succ.insert(prev.to_string(), i.clone());
        }
    }
    let closed = |i: &str| found.get(i).is_some_and(|m| m.valid_to.is_some());
    for i in ids {
        if closed(i) && !succ.contains_key(i) {
            if let Some(next) = found
                .get(i)
                .and_then(|m| m.superseded_by_id.as_deref())
                .filter(|s| !s.is_empty())
            {
                succ.insert(i.clone(), next.to_string());
            }
        }
    }
    let pooled: HashSet<&str> = ids.iter().map(String::as_str).collect();
    let head = |start: &str| -> Option<String> {
        let mut cur = start.to_string();
        let mut seen: HashSet<String> = HashSet::from([cur.clone()]);
        while let Some(next) = succ.get(&cur) {
            if !pooled.contains(next.as_str()) || seen.contains(next) {
                break;
            }
            cur = next.clone();
            seen.insert(cur.clone());
        }
        (seen.len() > 1).then_some(cur)
    };
    let pos: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(n, i)| (i.as_str(), n))
        .collect();
    let moving: Vec<&String> = ids
        .iter()
        .filter(|i| {
            closed(i)
                && head(i)
                    .and_then(|h| pos.get(h.as_str()).copied())
                    .is_some_and(|ph| pos[i.as_str()] < ph)
        })
        .collect();
    let moving_set: HashSet<&str> = moving.iter().map(|s| s.as_str()).collect();
    let mut out: Vec<String> = ids
        .iter()
        .filter(|i| !moving_set.contains(i.as_str()))
        .cloned()
        .collect();
    for i in moving {
        if let Some(h) = head(i) {
            if let Some(at) = out.iter().position(|x| *x == h) {
                out.insert(at + 1, i.clone());
            }
        }
    }
    out
}

/// Everything [`search`] needs besides the environment.
pub struct SearchInput<'a> {
    /// The raw query; [`search`] sanitizes it.
    pub query: &'a str,
    /// Who is asking. Required: there is no default, so a caller cannot forget
    /// the filter. Build it with [`ReaderScope::resolve`].
    pub scope: &'a ReaderScope,
    /// The page size.
    pub limit: usize,
    /// Classified arm outcomes, in the order the store lists them.
    pub arms: &'a [ArmOutcome],
    /// Hydrated candidates by id.
    pub found: &'a HashMap<String, Candidate>,
    /// A pg BM25 arm is attached (it is then the preferred lexical pin).
    pub has_pg_bm25: bool,
    /// Lexical index document frequency, when the primary offers one.
    pub df: Option<&'a dyn Fn(&str) -> i64>,
    /// Usage counters, `None` when the primary keeps none (decay is skipped).
    pub usage: Option<&'a HashMap<String, Usage>>,
    /// The clock decay is evaluated at.
    pub decay_as_of: Dt,
}

/// Each stage's output, for a trace in the style of `arm_diff`.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchTrace {
    pub route: Option<Route>,
    pub rerank_applied: bool,
    /// `fuse()` output (pool wide).
    pub fused: Vec<String>,
    /// After the rare-term reranker.
    pub reranked: Vec<String>,
    /// After supersession demotion, cut to the pool.
    pub demoted: Vec<String>,
    /// After the decay channel and the final demotion.
    pub decayed: Vec<String>,
    /// The page: tier cap then procedure order.
    pub page: Vec<String>,
}

/// Run the stages in the store's order. `None` means no arm produced a hit (the
/// Python `search_fused` returns `None` and `search` takes its legacy path).
pub fn search(input: &SearchInput, env: &Env) -> Option<SearchTrace> {
    let query_owned = sanitize_query(input.query);
    let query = query_owned.as_str();
    // The reader filter runs before anything is fused. For a restricted reader
    // an id that is not among the hydrated candidates has an unknowable
    // audience, so it is dropped too (fail closed).
    let scope = input.scope;
    let visible_found: HashMap<String, Candidate>;
    let visible_arms: Vec<ArmOutcome>;
    let (found, arms): (&HashMap<String, Candidate>, &[ArmOutcome]) = if scope.unrestricted() {
        (input.found, input.arms)
    } else {
        visible_found = input
            .found
            .iter()
            .filter(|(_, c)| scope.allows(Some(&c.context_tag), &c.tags))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        visible_arms = input
            .arms
            .iter()
            .map(|a| ArmOutcome {
                ids: a
                    .ids
                    .iter()
                    .filter(|i| visible_found.contains_key(*i))
                    .cloned()
                    .collect(),
                ..a.clone()
            })
            .collect();
        (&visible_found, &visible_arms)
    };
    let limit = input.limit;
    let pool = limit.max(limit * 3);
    let strategy = fusion::fusion_strategy(env);
    let route = (strategy == "pinned_rrf").then(|| fusion::route_query(query, input.df, None, env));
    let pin_arms: &[&str] = if input.has_pg_bm25 {
        &["lexical", "lexical_local"]
    } else {
        &["lexical_local", "lexical"]
    };
    let fused = fusion::fuse(
        arms,
        pool.max(DEFAULT_ARM_CAP),
        RRF_K,
        strategy,
        pin_arms,
        route.as_ref(),
        env,
    );
    let pool_ids: Vec<String> = fused
        .ids
        .iter()
        .filter(|i| found.contains_key(*i))
        .cloned()
        .collect();
    let mut rerank_applied = false;
    let mut reranked = pool_ids.clone();
    if !fused.ids.is_empty() {
        let texts: Vec<String> = pool_ids
            .iter()
            .map(|i| format!("{} {}", found[i].title, found[i].content))
            .collect();
        let outcome = Reranker::from_env(env).rerank(query, &texts);
        if outcome.applied() {
            rerank_applied = true;
            reranked = outcome.order.iter().map(|&i| pool_ids[i].clone()).collect();
        }
    }
    let mut demoted = demote_superseded(&reranked, found);
    demoted.truncate(pool);
    if demoted.is_empty() {
        return None;
    }
    let after_decay = match decay::configured_channel(env) {
        Some(ch) if demoted.len() >= 2 => decay::rerank(
            &demoted,
            found,
            input.usage,
            ch,
            input.decay_as_of,
            decay::RERANK_GAMMA,
        ),
        _ => demoted.clone(),
    };
    let decayed = demote_superseded(&after_decay, found);
    let capped = apply_tier_cap(&decayed, found, limit, env);
    let page = order_procedures(&capped, found, query);
    Some(SearchTrace {
        route: fused.route,
        rerank_applied,
        fused: fused.ids,
        reranked,
        demoted,
        decayed,
        page,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str) -> Candidate {
        Candidate {
            id: id.into(),
            title: id.into(),
            schema_version: 2,
            source_tier: Some("user-stated".into()),
            layer: "short-term".into(),
            kind: "fact".into(),
            ..Candidate::default()
        }
    }

    #[test]
    fn closed_record_moves_below_its_head() {
        let mut old = cand("old");
        old.valid_to = Some("2026-10-20T00:00:00+00:00".into());
        let mut new = cand("new");
        new.supersedes_id = Some("old".into());
        let found: HashMap<_, _> = [old, new, cand("x")]
            .into_iter()
            .map(|c| (c.id.clone(), c))
            .collect();
        let ids: Vec<String> = ["old", "x", "new"].iter().map(|s| s.to_string()).collect();
        assert_eq!(demote_superseded(&ids, &found), ["x", "new", "old"]);
    }
}
