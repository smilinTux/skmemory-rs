//! Query sanitizer: the Rust twin of `skmemory/query_sanitizer.py`
//! (SKMEM-R-06C).
//!
//! Agents sometimes prepend a whole system prompt to a search query, so the
//! embedding represents the prompt rather than the question. The cascade:
//! passthrough when short (200 chars or fewer), the last question sentence,
//! the last meaningful sentence, then a 500 char tail. All lengths and slices
//! are in characters, as in Python.

use crate::env::is_py_space;
use crate::model::py_strip;

const MAX_SHORT_QUERY: usize = 200;
const MAX_OUTPUT_LEN: usize = 500;

const SYSTEM_MARKERS: [&str; 9] = [
    "you are a",
    "you are an",
    "your role is",
    "system:",
    "instructions:",
    "## rules",
    "## context",
    "mandatory rules",
    "important:",
];

/// `sanitize_query`.
pub fn sanitize_query(raw: &str) -> String {
    if py_strip(raw).is_empty() {
        return String::new();
    }
    let query = py_strip(raw);
    if query.chars().count() <= MAX_SHORT_QUERY {
        return query.to_string();
    }
    if let Some(q) = extract_last_question(query).filter(|q| q.chars().count() >= 10) {
        return head(&q, MAX_OUTPUT_LEN);
    }
    if let Some(s) = extract_last_sentence(query).filter(|s| s.chars().count() >= 10) {
        return head(&s, MAX_OUTPUT_LEN);
    }
    let chars: Vec<char> = query.chars().collect();
    let from = chars.len().saturating_sub(MAX_OUTPUT_LEN);
    py_strip(&chars[from..].iter().collect::<String>()).to_string()
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn extract_last_question(text: &str) -> Option<String> {
    let c: Vec<char> = text.chars().collect();
    let q = c.iter().rposition(|&ch| ch == '?')?;
    let mut start = 0;
    for i in (0..q).rev() {
        if matches!(c[i], '.' | '!' | '?') && i + 1 < q {
            if c.get(i + 1) == Some(&' ') {
                start = i + 1;
                break;
            }
        } else if c[i] == '\n' {
            start = i + 1;
            break;
        }
    }
    let slice: String = c[start..=q].iter().collect();
    let result = strip_system_prefix(py_strip(&slice));
    (!result.is_empty()).then_some(result)
}

/// `re.split(r"(?<=[.!?])\s+", text)`.
fn split_sentences(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < c.len() {
        if i > 0 && matches!(c[i - 1], '.' | '!' | '?') && is_py_space(c[i]) {
            while i < c.len() && is_py_space(c[i]) {
                i += 1;
            }
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c[i]);
            i += 1;
        }
    }
    out.push(cur);
    out
}

fn extract_last_sentence(text: &str) -> Option<String> {
    for sentence in split_sentences(text).iter().rev() {
        let cleaned = py_strip(sentence);
        if cleaned.chars().count() >= 10 && !is_system_prompt_line(cleaned) {
            return Some(strip_system_prefix(cleaned));
        }
    }
    None
}

fn is_system_prompt_line(line: &str) -> bool {
    let lower = py_strip(&line.to_lowercase()).to_string();
    SYSTEM_MARKERS.iter().any(|m| lower.starts_with(m))
}

fn strip_system_prefix(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut skip = 0;
    while skip < lines.len() && is_system_prompt_line(lines[skip]) {
        skip += 1;
    }
    py_strip(&lines[skip..].join("\n")).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_passthrough_and_blank() {
        assert_eq!(sanitize_query("  hello  "), "hello");
        assert_eq!(sanitize_query("   \n"), "");
    }

    #[test]
    fn long_prompt_keeps_the_question() {
        let q = format!(
            "You are a helpful bot.\n{}\nWhat did we decide about the deploy?",
            "x ".repeat(200)
        );
        assert_eq!(sanitize_query(&q), "What did we decide about the deploy?");
    }
}
