//! Rare-term reranking for the fused pool: the port of `skmemory/rerank.py`.
//!
//! A query term is a boost candidate only when it looks like an identifier (it
//! carries a digit, or is at least 12 characters long) and is rare in the pool
//! (at most 15 percent of the documents). Boosted candidates move up in RANK
//! space (`rank / (divisor * hits)`, never below 1), which is bounded: a
//! candidate already at rank 1 cannot be displaced. With no such term the
//! reranker abstains and returns the identity order.

use crate::env::{casefold, Env};
use std::collections::HashSet;

/// A term must appear in at most this fraction of the pool to discriminate.
pub const DEFAULT_MAX_DF_RATIO: f64 = 0.15;
/// Rank divisor for a boosted candidate.
pub const DEFAULT_RANK_DIVISOR: f64 = 5.0;
/// Minimum length for a digit-free token to count as identifier-shaped.
pub const DEFAULT_MIN_OPAQUE_LEN: usize = 12;

/// `[a-z0-9]+` runs of the casefolded text, at least two characters long.
pub fn tokenize(text: &str) -> Vec<String> {
    let folded = casefold(text);
    let b = folded.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_lowercase() || b[i].is_ascii_digit() {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_lowercase() || b[j].is_ascii_digit()) {
                j += 1;
            }
            if j - i >= 2 {
                out.push(folded[i..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// A digit is the strong signal; a long opaque token also qualifies.
pub fn identifier_shaped(term: &str, min_opaque_len: usize) -> bool {
    term.bytes().any(|c| c.is_ascii_digit()) || term.chars().count() >= min_opaque_len
}

/// Identifier-shaped query terms rare enough to separate pool members, most
/// discriminative first. Empty means the reranker should abstain.
pub fn discriminative_terms(
    query: &str,
    documents: &[String],
    max_df_ratio: f64,
    min_opaque_len: usize,
) -> Vec<String> {
    if documents.is_empty() {
        return Vec::new();
    }
    let doc_tokens: Vec<HashSet<String>> = documents
        .iter()
        .map(|d| tokenize(d).into_iter().collect())
        .collect();
    let n = doc_tokens.len();
    let cap = ((max_df_ratio * n as f64) as usize).max(1);
    let mut seen = HashSet::new();
    let mut scored: Vec<(String, f64)> = Vec::new();
    for term in tokenize(query) {
        if !seen.insert(term.clone()) || !identifier_shaped(&term, min_opaque_len) {
            continue;
        }
        let df = doc_tokens.iter().filter(|t| t.contains(&term)).count();
        if df == 0 || df > cap {
            continue;
        }
        scored.push((term, ((n + 1) as f64 / (df as f64 + 0.5)).ln()));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().map(|(t, _)| t).collect()
}

/// Whether the reranker changed anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerankStatus {
    Applied,
    Abstained,
}

impl RerankStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RerankStatus::Applied => "applied",
            RerankStatus::Abstained => "abstained",
        }
    }
}

/// The new order (indexes into the candidate list) and why.
#[derive(Clone, Debug, PartialEq)]
pub struct RerankOutcome {
    pub order: Vec<usize>,
    pub status: RerankStatus,
    pub boost_terms: Vec<String>,
    pub boosted: Vec<usize>,
}

impl RerankOutcome {
    pub fn applied(&self) -> bool {
        self.status == RerankStatus::Applied
    }
}

/// The reranker the store builds from the environment.
#[derive(Clone, Debug, PartialEq)]
pub enum Reranker {
    /// `SKMEMORY_SEARCH_RERANK=off`: does nothing, and says so.
    Noop,
    RareTerm {
        rank_divisor: f64,
        max_df_ratio: f64,
        min_opaque_len: usize,
    },
}

impl Reranker {
    /// `MemoryStore._reranker`: `SKMEMORY_SEARCH_RERANK` (`on` default; `off`,
    /// `0`, `false`, `no` disable) and `SKMEMORY_SEARCH_RERANK_DIVISOR` (a
    /// float; garbage or 0 keeps the default).
    pub fn from_env(env: &Env) -> Reranker {
        let mode = casefold(env.get_or("SKMEMORY_SEARCH_RERANK", "on").trim());
        if matches!(mode.as_str(), "off" | "0" | "false" | "no") {
            return Reranker::Noop;
        }
        let raw = env.get_or("SKMEMORY_SEARCH_RERANK_DIVISOR", "").trim();
        let divisor = if raw.is_empty() {
            None
        } else {
            raw.parse::<f64>().ok()
        };
        Reranker::RareTerm {
            rank_divisor: divisor
                .filter(|d| *d != 0.0)
                .unwrap_or(DEFAULT_RANK_DIVISOR),
            max_df_ratio: DEFAULT_MAX_DF_RATIO,
            min_opaque_len: DEFAULT_MIN_OPAQUE_LEN,
        }
    }

    /// Reorder `documents` (already ranked) for `query`.
    pub fn rerank(&self, query: &str, documents: &[String]) -> RerankOutcome {
        let identity: Vec<usize> = (0..documents.len()).collect();
        let abstain = |order: Vec<usize>| RerankOutcome {
            order,
            status: RerankStatus::Abstained,
            boost_terms: Vec::new(),
            boosted: Vec::new(),
        };
        let Reranker::RareTerm {
            rank_divisor,
            max_df_ratio,
            min_opaque_len,
        } = self
        else {
            return abstain(identity);
        };
        if documents.len() < 2 {
            return abstain(identity);
        }
        let terms = discriminative_terms(query, documents, *max_df_ratio, *min_opaque_len);
        if terms.is_empty() {
            return abstain(identity);
        }
        let term_set: HashSet<&str> = terms.iter().map(String::as_str).collect();
        let divisor = rank_divisor.max(1.0);
        let mut keys: Vec<(f64, usize)> = Vec::with_capacity(documents.len());
        let mut boosted = Vec::new();
        for (idx, doc) in documents.iter().enumerate() {
            let toks: HashSet<String> = tokenize(doc).into_iter().collect();
            let rank = idx + 1;
            let hits = term_set.iter().filter(|t| toks.contains(**t)).count();
            if hits > 0 {
                boosted.push(idx);
                keys.push(((rank as f64 / (divisor * hits as f64)).max(1.0), idx));
            } else {
                keys.push((rank as f64, idx));
            }
        }
        keys.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        RerankOutcome {
            order: keys.into_iter().map(|(_, i)| i).collect(),
            status: RerankStatus::Applied,
            boost_terms: terms,
            boosted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_drops_short_and_punctuation() {
        assert_eq!(tokenize("A prb-6f069c5e x1 Z"), ["prb", "6f069c5e", "x1"]);
    }

    #[test]
    fn identifier_boost_is_rank_bounded() {
        let docs: Vec<String> = ["alpha beta", "gamma delta", "zioap01 note", "pad pad"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r = Reranker::from_env(&Env::empty()).rerank("zioap01", &docs);
        assert!(r.applied());
        // rank 3 / 5 floors at 1.0: it ties the head and cannot displace it
        assert_eq!(r.order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn spread_term_abstains() {
        let docs: Vec<String> = (0..6).map(|i| format!("common zioap01 {i}")).collect();
        let r = Reranker::from_env(&Env::empty()).rerank("zioap01", &docs);
        assert!(!r.applied());
        assert_eq!(r.order, vec![0, 1, 2, 3, 4, 5]);
    }
}
