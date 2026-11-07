//! A snapshot of the environment variables the retrieval stages read.
//!
//! Python reads `os.environ` at call time. Rust takes an explicit [`Env`] so a
//! test can run cases in parallel with different settings and a caller can pin
//! one search to one set of settings. [`Env::process`] captures the live
//! process environment (only `SKMEMORY_*` names are kept).

use std::collections::HashMap;

/// Environment variable values by name. An unset variable is absent; a variable
/// set to the empty string is present and empty (Python `environ.get` sees both).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Env(HashMap<String, String>);

impl Env {
    /// No variables set.
    pub fn empty() -> Env {
        Env::default()
    }

    /// The live process environment, `SKMEMORY_*` variables only.
    pub fn process() -> Env {
        Env(std::env::vars()
            .filter(|(k, _)| k.starts_with("SKMEMORY_"))
            .collect())
    }

    /// Build from `(name, value)` pairs.
    pub fn from_pairs<K: Into<String>, V: Into<String>>(
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Env {
        Env(pairs
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect())
    }

    /// The value of `name`, if set.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    /// `os.environ.get(name, default)`.
    pub fn get_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.get(name).unwrap_or(default)
    }

    /// A copy with `name` set.
    pub fn with(mut self, name: &str, value: &str) -> Env {
        self.0.insert(name.to_string(), value.to_string());
        self
    }
}

/// Python `str.casefold` for the characters retrieval cares about: full
/// lowercase plus the multi-letter folds Python applies (`ß`, the `ﬀ`..`ﬆ`
/// ligatures, long s). Other characters are lowercased.
pub(crate) fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'ß' | 'ẞ' => out.push_str("ss"),
            'ſ' => out.push('s'),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' | 'ﬆ' => out.push_str("st"),
            _ => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// Python `str.isspace` for one character (`\s` in a `str` pattern).
pub(crate) fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_vs_empty() {
        let e = Env::from_pairs([("SKMEMORY_X", "")]);
        assert_eq!(e.get("SKMEMORY_X"), Some(""));
        assert_eq!(e.get("SKMEMORY_Y"), None);
        assert_eq!(e.get_or("SKMEMORY_Y", "d"), "d");
    }

    #[test]
    fn casefold_expansions() {
        assert_eq!(casefold("StraSSe Straße ﬁx"), "strasse strasse fix");
    }
}
