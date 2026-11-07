//! Failure procedures never outrank successful ones for a how-to query: the
//! port of `skmemory/procedures.py` (`order_procedures`).
//!
//! The two Python regexes are matched by a small backtracking matcher over the
//! same alternations (case-insensitive, Unicode `\s` and `\w`, `\b`).

use crate::env::is_py_space;
use crate::search::Candidate;
use crate::util::is_py_word_char;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum El {
    /// A lowercase literal, matched case-insensitively.
    Lit(&'static str),
    /// `\s+`
    Space,
    /// `'?`
    OptApos,
    /// `\w*`
    WordStar,
    /// `\b`
    Boundary,
}
use El::{Boundary as B, Lit as L, OptApos, Space as S, WordStar as W};

/// `how\s+(do|can|should|to|would)\b | steps?\s+(to|for)\b | procedure | runbook
/// | instructions?\s+(to|for) | way\s+to | recover | restore | fix`, each
/// reached after a leading `\b`. (Only the first three groups end in `\b`.)
const HOWTO: &[&[El]] = &[
    &[L("how"), S, L("do"), B],
    &[L("how"), S, L("can"), B],
    &[L("how"), S, L("should"), B],
    &[L("how"), S, L("to"), B],
    &[L("how"), S, L("would"), B],
    &[L("steps"), S, L("to"), B],
    &[L("steps"), S, L("for"), B],
    &[L("step"), S, L("to"), B],
    &[L("step"), S, L("for"), B],
    &[L("procedure")],
    &[L("runbook")],
    &[L("instructions"), S, L("to")],
    &[L("instructions"), S, L("for")],
    &[L("instruction"), S, L("to")],
    &[L("instruction"), S, L("for")],
    &[L("way"), S, L("to")],
    &[L("recover")],
    &[L("restore")],
    &[L("fix")],
];

/// `\b(fail\w*|warning|avoid|mistake|went\s+wrong|broke\w*|should\s+i|
/// shouldn'?t|don'?t|do\s+not|why\s+did|didn'?t\s+work|not\s+work\w*|risk\w*|
/// danger\w*)\b`
const FAILURE_ASK: &[&[El]] = &[
    &[L("fail"), W, B],
    &[L("warning"), B],
    &[L("avoid"), B],
    &[L("mistake"), B],
    &[L("went"), S, L("wrong"), B],
    &[L("broke"), W, B],
    &[L("should"), S, L("i"), B],
    &[L("shouldn"), OptApos, L("t"), B],
    &[L("don"), OptApos, L("t"), B],
    &[L("do"), S, L("not"), B],
    &[L("why"), S, L("did"), B],
    &[L("didn"), OptApos, L("t"), S, L("work"), B],
    &[L("not"), S, L("work"), W, B],
    &[L("risk"), W, B],
    &[L("danger"), W, B],
];

fn lower(c: char) -> char {
    match c {
        'ſ' => 's',
        _ => c.to_lowercase().next().unwrap_or(c),
    }
}

fn is_word(c: char) -> bool {
    is_py_word_char(c)
}

fn at_boundary(t: &[char], p: usize) -> bool {
    let before = p > 0 && is_word(t[p - 1]);
    let after = p < t.len() && is_word(t[p]);
    before != after
}

fn matches_at(els: &[El], t: &[char], p: usize) -> bool {
    let Some((first, rest)) = els.split_first() else {
        return true;
    };
    match *first {
        El::Lit(s) => {
            let mut q = p;
            for ch in s.chars() {
                if q >= t.len() || lower(t[q]) != ch {
                    return false;
                }
                q += 1;
            }
            matches_at(rest, t, q)
        }
        El::Space => {
            let n = t[p..].iter().take_while(|c| is_py_space(**c)).count();
            (1..=n).rev().any(|k| matches_at(rest, t, p + k))
        }
        El::WordStar => {
            let n = t[p..].iter().take_while(|c| is_word(**c)).count();
            (0..=n).rev().any(|k| matches_at(rest, t, p + k))
        }
        El::OptApos => (p < t.len() && t[p] == '\'' && matches_at(rest, t, p + 1))
            || matches_at(rest, t, p),
        El::Boundary => at_boundary(t, p) && matches_at(rest, t, p),
    }
}

/// `re.search(r"\b(alt1|alt2...)", text)`.
fn search(alts: &[&[El]], text: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    (0..=t.len()).any(|p| at_boundary(&t, p) && alts.iter().any(|a| matches_at(a, &t, p)))
}

/// A how-to query that does not ask about failure.
pub fn is_howto_query(query: &str) -> bool {
    search(HOWTO, query) && !search(FAILURE_ASK, query)
}

fn is_failure_procedure(c: &Candidate) -> bool {
    c.kind == "procedure" && c.outcome.as_deref() == Some("failure")
}

/// Procedure slots are refilled successes first, then failures (stable). Other
/// items keep their place. Unchanged for a query that is not a plain how-to.
pub fn order_procedures(
    ids: &[String],
    found: &HashMap<String, Candidate>,
    query: &str,
) -> Vec<String> {
    if !is_howto_query(query) {
        return ids.to_vec();
    }
    let proc = |i: &String| found.get(i).is_some_and(|c| c.kind == "procedure");
    let slots: Vec<usize> = (0..ids.len()).filter(|&i| proc(&ids[i])).collect();
    let procs: Vec<&String> = slots.iter().map(|&i| &ids[i]).collect();
    let fail = |i: &String| found.get(i).is_some_and(is_failure_procedure);
    let ordered: Vec<&String> = procs
        .iter()
        .filter(|i| !fail(i))
        .chain(procs.iter().filter(|i| fail(i)))
        .copied()
        .collect();
    let mut out = ids.to_vec();
    for (slot, id) in slots.into_iter().zip(ordered) {
        out[slot] = id.clone();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn howto_and_failure_shapes() {
        assert!(is_howto_query("how do I restore the replica"));
        assert!(is_howto_query("fixture setup"));
        assert!(is_howto_query("way tomorrow"));
        assert!(!is_howto_query("how to avoid a mistake"));
        assert!(!is_howto_query("should I restore"));
        assert!(!is_howto_query("plain query"));
        assert!(!is_howto_query("howdy"));
        assert!(is_howto_query("how\tto"));
    }
}
