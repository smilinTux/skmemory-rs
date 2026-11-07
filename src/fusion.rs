//! Multi-arm retrieval fusion: the port of `skmemory/fusion.py` (SKMEM-R-06).
//!
//! Arms are run by the caller; this module classifies each arm's outcome,
//! routes the query (identifier or natural language), and merges the arms:
//!
//! * [`run_arm`] classifies an arm (`ok`, `abstained`, `unavailable`, `failed`,
//!   `disabled`), drops empty ids, de-duplicates inside the arm and caps it at
//!   [`DEFAULT_ARM_CAP`];
//! * [`route_query`] is the deterministic shape router;
//! * [`reciprocal_rank_fusion`], [`rank_interleave`] and [`pinned_rrf`] are the
//!   three mergers, and [`fuse`] picks one and applies the route's pins.
//!
//! Environment settings come in through an [`Env`] (Python reads `os.environ`):
//! `SKMEMORY_FUSION_STRATEGY`, `_PIN`, `_VECTOR_PIN`, `_ROUTE`, `_FILL`.
//!
//! Not ported here: the thread pool and the arm guard (the arms are the
//! caller's), and the temporal window parser (the temporal arm is the caller's
//! to run; its outcome enters [`fuse`] like any other arm).

use crate::env::Env;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Reciprocal Rank Fusion constant (Cormack et al.).
pub const RRF_K: i64 = 60;
/// Per-arm candidate cap applied before the merge.
pub const DEFAULT_ARM_CAP: usize = 25;
/// Default number of pinned lexical hits.
pub const DEFAULT_PIN: usize = 3;
/// Lexical arms in pin-preference order.
pub const LEXICAL_ARMS: [&str; 2] = ["lexical", "lexical_local"];
/// Identifier route pin depth.
pub const IDENTIFIER_PIN: usize = 3;
/// Identifier route pin arms: the local FTS5 arm first, even with pg attached.
pub const IDENTIFIER_PIN_ARMS: [&str; 2] = ["lexical_local", "lexical"];
/// Natural route default vector pin depth.
pub const DEFAULT_VECTOR_PIN: usize = 4;
/// Arms that can fill the single lexical slot on the natural route.
pub const NATURAL_LEXICAL_SLOT_ARMS: [&str; 2] = ["lexical_local", "lexical"];
/// A token with document frequency in `1..=RARE_DF_MAX` marks an identifier query.
pub const RARE_DF_MAX: i64 = 3;
/// The dense arm's name.
pub const VECTOR_ARM: &str = "semantic";

pub const STRATEGY_ENV: &str = "SKMEMORY_FUSION_STRATEGY";
pub const ROUTE_ENV: &str = "SKMEMORY_FUSION_ROUTE";
pub const VECTOR_PIN_ENV: &str = "SKMEMORY_FUSION_VECTOR_PIN";
pub const PIN_ENV: &str = "SKMEMORY_FUSION_PIN";
pub const FILL_ENV: &str = "SKMEMORY_FUSION_FILL";

/// What an arm did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArmStatus {
    Ok,
    Abstained,
    Unavailable,
    Failed,
    Disabled,
    Timeout,
}

impl ArmStatus {
    /// The Python status string.
    pub fn as_str(self) -> &'static str {
        match self {
            ArmStatus::Ok => "ok",
            ArmStatus::Abstained => "abstained",
            ArmStatus::Unavailable => "unavailable",
            ArmStatus::Failed => "failed",
            ArmStatus::Disabled => "disabled",
            ArmStatus::Timeout => "timeout",
        }
    }
}

/// One arm's classified outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmOutcome {
    pub name: String,
    pub ids: Vec<String>,
    pub status: ArmStatus,
    pub detail: String,
}

impl ArmOutcome {
    /// `status == ok` and at least one id.
    pub fn contributed(&self) -> bool {
        self.status == ArmStatus::Ok && !self.ids.is_empty()
    }
}

/// What the caller's arm produced, before classification.
#[derive(Clone, Debug)]
pub enum ArmInput {
    /// The arm ran and returned these ids (best first). May be empty.
    Ids(Vec<String>),
    /// The backing store is not attached.
    Unattached,
    /// The arm raised.
    Failed(String),
    /// Switched off by configuration, with the reason.
    Disabled(String),
    /// Did not answer before the deadline.
    TimedOut(String),
}

/// `cap_per_source`: the best `cap` entries (`cap == 0` keeps all).
pub fn cap_per_source(ids: &[String], cap: usize) -> Vec<String> {
    if cap == 0 || ids.len() <= cap {
        ids.to_vec()
    } else {
        ids[..cap].to_vec()
    }
}

/// `run_arm`: classify one arm instead of swallowing its outcome.
pub fn run_arm(name: &str, input: ArmInput, cap: usize) -> ArmOutcome {
    let mk = |status, detail: &str, ids: Vec<String>| ArmOutcome {
        name: name.to_string(),
        ids,
        status,
        detail: detail.to_string(),
    };
    match input {
        ArmInput::Disabled(why) => mk(ArmStatus::Disabled, &why, vec![]),
        ArmInput::Unattached => mk(ArmStatus::Unavailable, "backend not attached", vec![]),
        ArmInput::Failed(msg) => mk(ArmStatus::Failed, &msg, vec![]),
        ArmInput::TimedOut(msg) => mk(ArmStatus::Timeout, &msg, vec![]),
        ArmInput::Ids(raw) => {
            let ids: Vec<String> = raw.into_iter().filter(|i| !i.is_empty()).collect();
            if ids.is_empty() {
                return mk(ArmStatus::Abstained, "", vec![]);
            }
            let mut seen = HashSet::new();
            let ordered: Vec<String> = ids
                .into_iter()
                .filter(|i| seen.insert(i.clone()))
                .collect();
            mk(ArmStatus::Ok, "", cap_per_source(&ordered, cap))
        }
    }
}

/// The merger the environment selects.
pub fn fusion_strategy(env: &Env) -> &'static str {
    let raw = crate::env::casefold(env.get_or(STRATEGY_ENV, "").trim()).replace('-', "_");
    match raw.as_str() {
        "interleave" | "rank_interleave" | "old" | "legacy" => "interleave",
        "rrf" | "plain_rrf" => "rrf",
        _ => "pinned_rrf",
    }
}

fn pin_from(env: &Env, name: &str, default: usize) -> usize {
    let raw = env.get_or(name, "").trim();
    if raw.is_empty() {
        return default;
    }
    match raw.parse::<i128>() {
        Ok(n) => n.max(0).min(usize::MAX as i128) as usize,
        Err(_) => default,
    }
}

/// `fusion_pin`: lexical hits the default fusion pins (`SKMEMORY_FUSION_PIN`).
pub fn fusion_pin(env: &Env) -> usize {
    pin_from(env, PIN_ENV, DEFAULT_PIN)
}

/// `fusion_vector_pin`: vector hits the natural route pins (default 4).
pub fn fusion_vector_pin(env: &Env) -> usize {
    pin_from(env, VECTOR_PIN_ENV, DEFAULT_VECTOR_PIN)
}

/// `fusion_route_mode`: `auto`, `lexical` or `vector`.
pub fn fusion_route_mode(env: &Env) -> &'static str {
    match crate::env::casefold(env.get_or(ROUTE_ENV, "auto").trim()).as_str() {
        "lexical" => "lexical",
        "vector" => "vector",
        _ => "auto",
    }
}

/// The class a query was routed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Identifier,
    Natural,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Identifier => "identifier",
            Route::Natural => "natural",
        }
    }
}

/// Which head the fusion pins, and the shape rule that decided it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteDecision {
    pub cls: Route,
    pub reason: &'static str,
}

const QUESTION_LEADS: &[&str] = &[
    "who", "whom", "whose", "what", "when", "where", "why", "how", "which", "did", "do", "does",
    "is", "are", "was", "were", "can", "could", "should", "would", "will", "have", "has", "had",
];
const QUESTION_LEADS_EXTRA: &[&str] = &["tell", "explain", "describe", "remind"];
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "about", "from", "by", "and",
    "or", "but", "if", "then", "so", "as", "it", "its", "this", "that", "these", "those", "there",
    "here", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them", "his", "her",
    "their", "be", "been", "being", "am", "is", "are", "was", "were", "do", "does", "did", "have",
    "has", "had", "not", "no", "what", "when", "where", "why", "how", "which", "who", "whom",
    "can",
];
const STOPWORDS_TAIL: &[&str] = &["could", "should", "would", "will", "just"];
const STOPWORDS_LAST: &[&str] = &["any", "some"];

fn is_question_lead(t: &str) -> bool {
    QUESTION_LEADS.contains(&t) || QUESTION_LEADS_EXTRA.contains(&t)
}

fn is_stopword(t: &str) -> bool {
    STOPWORDS.contains(&t) || STOPWORDS_TAIL.contains(&t) || STOPWORDS_LAST.contains(&t)
}

/// `[a-z0-9][a-z0-9._\-]*` over the query (ASCII), each match stripped of
/// `._-` and lowercased.
fn route_tokens(query: &str) -> Vec<String> {
    let b = query.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphanumeric() {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || matches!(b[j], b'.' | b'_' | b'-'))
            {
                j += 1;
            }
            let tok = query[i..j].trim_matches(|c| matches!(c, '.' | '_' | '-'));
            if !tok.is_empty() {
                out.push(tok.to_ascii_lowercase());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// `_DATE_TOKEN_RE`: `20dd`, `20dd-dd`, `20dd-dd-dd`, or 1-2 digits with an
/// optional ordinal suffix.
fn is_date_token(t: &str) -> bool {
    let d = |s: &str, n: usize| s.len() == n && s.bytes().all(|c| c.is_ascii_digit());
    let year = |s: &str| s.len() == 4 && s.starts_with("20") && d(s, 4);
    let parts: Vec<&str> = t.split('-').collect();
    match parts.as_slice() {
        [y] if year(y) => return true,
        [y, m] if year(y) && d(m, 2) => return true,
        [y, m, dd] if year(y) && d(m, 2) && d(dd, 2) => return true,
        _ => {}
    }
    let digits = t.bytes().take_while(u8::is_ascii_digit).count();
    if !(1..=2).contains(&digits) {
        return false;
    }
    matches!(&t[digits..], "" | "st" | "nd" | "rd" | "th")
}

/// `_HOST_RE`: dot-separated labels, each `[a-z0-9][a-z0-9-]*`, at least two.
fn is_host(t: &str) -> bool {
    let parts: Vec<&str> = t.split('.').collect();
    parts.len() >= 2
        && parts.iter().all(|p| {
            let b = p.as_bytes();
            !b.is_empty()
                && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
                && b[1..]
                    .iter()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
        })
}

fn strong_identifier(tok: &str) -> bool {
    if is_date_token(tok) {
        return false;
    }
    tok.bytes().any(|c| c.is_ascii_digit()) || tok.contains('_') || is_host(tok)
}

/// `route_query`: classify `query` as identifier-ish or natural language.
///
/// `df` is the optional lexical-index document frequency. `mode` overrides the
/// environment (`SKMEMORY_FUSION_ROUTE`) when given.
pub fn route_query(
    query: &str,
    df: Option<&dyn Fn(&str) -> i64>,
    mode: Option<&str>,
    env: &Env,
) -> RouteDecision {
    let mode = match mode {
        Some(m) if !m.is_empty() => m,
        _ => fusion_route_mode(env),
    };
    let dec = |cls, reason| RouteDecision { cls, reason };
    if mode == "lexical" {
        return dec(Route::Identifier, "forced");
    }
    if mode == "vector" {
        return dec(Route::Natural, "forced");
    }
    let toks = route_tokens(query);
    if toks.is_empty() {
        return dec(Route::Identifier, "empty");
    }
    if toks.iter().any(|t| strong_identifier(t)) {
        return dec(Route::Identifier, "id-shape");
    }
    if let Some(df) = df {
        for tok in toks.iter().filter(|t| !is_stopword(t)).take(8) {
            let n = df(tok);
            if (1..=RARE_DF_MAX).contains(&n) {
                return dec(Route::Identifier, "rare-token");
            }
        }
    }
    if is_question_lead(&toks[0]) || query.trim_end().ends_with('?') {
        return dec(Route::Natural, "question");
    }
    let stop = toks.iter().filter(|t| is_stopword(t)).count();
    if toks.len() >= 4 && stop as f64 / toks.len() as f64 >= 1.0 / 3.0 {
        return dec(Route::Natural, "stopwords");
    }
    dec(Route::Identifier, "keywords")
}

/// A fused candidate: id, RRF score, and the rank it held in each arm.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub id: String,
    pub score: f64,
    pub prov: BTreeMap<String, usize>,
}

/// Arms that voted, in arm order (a Python dict: a repeated name keeps its
/// first position and its last value).
pub type ArmResults = Vec<(String, Vec<String>)>;

fn arm_get<'a>(arms: &'a ArmResults, name: &str) -> Option<&'a Vec<String>> {
    arms.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// Classic RRF: `score(d) = sum over arms of 1 / (k + rank(d))`, best first,
/// ties broken by id so the order is deterministic.
pub fn reciprocal_rank_fusion(arms: &ArmResults, k: i64) -> Vec<Scored> {
    let mut by_id: HashMap<&str, Scored> = HashMap::new();
    for (arm, ids) in arms {
        for (i, id) in ids.iter().enumerate() {
            let rank = i + 1;
            let e = by_id.entry(id).or_insert_with(|| Scored {
                id: id.clone(),
                score: 0.0,
                prov: BTreeMap::new(),
            });
            e.score += 1.0 / (k + rank as i64) as f64;
            e.prov.insert(arm.clone(), rank);
        }
    }
    let mut out: Vec<Scored> = by_id.into_values().collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    out
}

/// RRF with each arm's champion protected: sort by best rank in any arm, then
/// RRF score, then id.
pub fn rank_interleave(arms: &ArmResults, k: i64) -> Vec<Scored> {
    let mut best: HashMap<&str, usize> = HashMap::new();
    for (_, ids) in arms {
        for (i, id) in ids.iter().enumerate() {
            let e = best.entry(id).or_insert(i + 1);
            *e = (*e).min(i + 1);
        }
    }
    let mut out = reciprocal_rank_fusion(arms, k);
    out.sort_by(|a, b| {
        best[a.id.as_str()]
            .cmp(&best[b.id.as_str()])
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

/// Pin the head of the strongest arm, then fill.
///
/// The pinned arm is the first of `pin_arms` that contributed, `pin` of its top
/// hits go first (`None` reads `SKMEMORY_FUSION_PIN`). `slot_arms` reserves one
/// slot after the head for the first hit of the first of those arms that is not
/// already in it. The fill is [`rank_interleave`], or plain RRF when
/// `SKMEMORY_FUSION_FILL=rrf`.
pub fn pinned_rrf(
    arms: &ArmResults,
    k: i64,
    pin: Option<usize>,
    pin_arms: &[&str],
    slot_arms: &[&str],
    env: &Env,
) -> Vec<Scored> {
    let pin = pin.unwrap_or_else(|| fusion_pin(env));
    let fill_rrf = crate::env::casefold(env.get_or(FILL_ENV, "interleave").trim()) == "rrf";
    let fused = if fill_rrf {
        reciprocal_rank_fusion(arms, k)
    } else {
        rank_interleave(arms, k)
    };
    let by_id: HashMap<&str, &Scored> = fused.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut head: Vec<String> = Vec::new();
    for name in pin_arms {
        if let Some(ids) = arm_get(arms, name).filter(|v| !v.is_empty()) {
            head = ids.iter().take(pin).cloned().collect();
            break;
        }
    }
    for name in slot_arms {
        let extra = arm_get(arms, name)
            .and_then(|v| v.iter().find(|d| !head.contains(d)))
            .cloned();
        if let Some(extra) = extra {
            head.push(extra);
            break;
        }
    }
    let pinned: HashSet<&str> = head.iter().map(String::as_str).collect();
    let mut out: Vec<Scored> = head.iter().map(|d| by_id[d.as_str()].clone()).collect();
    out.extend(
        fused
            .iter()
            .filter(|s| !pinned.contains(s.id.as_str()))
            .cloned(),
    );
    out
}

/// Fused ranking plus the provenance needed to explain it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FusionResult {
    pub ids: Vec<String>,
    pub arms: Vec<ArmOutcome>,
    /// doc id -> arm name -> rank within that arm
    pub provenance: BTreeMap<String, BTreeMap<String, usize>>,
    /// The route class, `None` when unrouted (or nothing contributed).
    pub route: Option<Route>,
}

/// Merge executed arms into one ranking.
///
/// Only arms with `ok` status and ids vote. `strategy` is `pinned_rrf`,
/// `interleave` or anything else for plain RRF. A route of `identifier` pins the
/// local lexical head (depth `max(pin, 3)`) when a lexical arm contributed;
/// `natural` pins the vector head (`SKMEMORY_FUSION_VECTOR_PIN`, default 4) plus
/// one lexical slot when the vector arm contributed, else the caller's pin stands.
pub fn fuse(
    arms: &[ArmOutcome],
    limit: usize,
    k: i64,
    strategy: &str,
    pin_arms: &[&str],
    route: Option<&RouteDecision>,
    env: &Env,
) -> FusionResult {
    let mut contributing: ArmResults = Vec::new();
    for a in arms.iter().filter(|a| a.contributed()) {
        match contributing.iter_mut().find(|(n, _)| *n == a.name) {
            Some(slot) => slot.1 = a.ids.clone(),
            None => contributing.push((a.name.clone(), a.ids.clone())),
        }
    }
    if contributing.is_empty() {
        return FusionResult {
            arms: arms.to_vec(),
            ..FusionResult::default()
        };
    }
    let has = |n: &str| contributing.iter().any(|(m, _)| m == n);
    let mut pin_arms: Vec<&str> = pin_arms.to_vec();
    let mut pin: Option<usize> = None;
    let mut slot_arms: Vec<&str> = Vec::new();
    if let Some(r) = route {
        if r.cls == Route::Identifier && IDENTIFIER_PIN_ARMS.iter().any(|n| has(n)) {
            pin_arms = IDENTIFIER_PIN_ARMS.to_vec();
            pin = Some(fusion_pin(env).max(IDENTIFIER_PIN));
        }
        if r.cls == Route::Natural && has(VECTOR_ARM) {
            pin_arms = vec![VECTOR_ARM];
            pin = Some(fusion_vector_pin(env));
            slot_arms = NATURAL_LEXICAL_SLOT_ARMS.to_vec();
        }
    }
    let ranked = match strategy {
        "pinned_rrf" => pinned_rrf(&contributing, k, pin, &pin_arms, &slot_arms, env),
        "interleave" => rank_interleave(&contributing, k),
        _ => reciprocal_rank_fusion(&contributing, k),
    };
    let top: Vec<&Scored> = ranked.iter().take(limit).collect();
    FusionResult {
        ids: top.iter().map(|s| s.id.clone()).collect(),
        arms: arms.to_vec(),
        provenance: top
            .iter()
            .map(|s| (s.id.clone(), s.prov.clone()))
            .collect(),
        route: route.map(|r| r.cls),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn run_arm_classifies_dedupes_and_caps() {
        let o = run_arm("x", ArmInput::Ids(ids(&["a", "", "b", "a"])), 25);
        assert_eq!((o.status, o.ids), (ArmStatus::Ok, ids(&["a", "b"])));
        assert_eq!(
            run_arm("x", ArmInput::Ids(ids(&["", ""])), 25).status,
            ArmStatus::Abstained
        );
        let long: Vec<String> = (0..40).map(|i| format!("m{i}")).collect();
        assert_eq!(run_arm("x", ArmInput::Ids(long), 25).ids.len(), 25);
        assert_eq!(
            run_arm("x", ArmInput::Unattached, 25).status,
            ArmStatus::Unavailable
        );
    }

    #[test]
    fn date_and_host_shapes() {
        for t in ["2026", "2026-10", "2026-10-16", "5", "12th", "1st"] {
            assert!(is_date_token(t), "{t}");
        }
        for t in ["20260", "2026-1", "123", "5x", "prb-6f069c5e"] {
            assert!(!is_date_token(t), "{t}");
        }
        assert!(is_host("a.b.c") && is_host("grafana.example.net"));
        assert!(!is_host("a..b") && !is_host("abc") && !is_host("a.-b"));
    }

    #[test]
    fn natural_route_pins_vector_head_and_one_lexical_slot() {
        let arms = vec![
            run_arm("semantic", ArmInput::Ids(ids(&["a", "b", "c", "d", "e"])), 25),
            run_arm("lexical_local", ArmInput::Ids(ids(&["a", "x", "y"])), 25),
        ];
        let r = RouteDecision {
            cls: Route::Natural,
            reason: "t",
        };
        let f = fuse(&arms, 10, RRF_K, "pinned_rrf", &LEXICAL_ARMS, Some(&r), &Env::empty());
        assert_eq!(&f.ids[..5], &ids(&["a", "b", "c", "d", "x"])[..]);
    }
}
