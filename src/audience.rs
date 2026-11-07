//! The reader (audience) ceiling: the Rust twin of `skmemory/audience.py`
//! `ReaderScope` / `resolve_reader` (SKMEM-R-06C).
//!
//! A [`ReaderScope`] says which memories a declared reader may receive: the
//! exact set of `context_tag` values it may see (or `None` for the owner) plus
//! the never-share tag exclusions. Every public entry point that can return a
//! memory takes a `&ReaderScope`, so a caller cannot forget the filter.
//!
//! Security properties, each covered by a test:
//!
//! - `SKMEMORY_READER` is a CEILING. A per-call reader can only narrow it.
//! - An unparseable reader is `@public`. It never widens access.
//! - There is no public constructor that skips the environment ceiling: every
//!   scope comes from [`ReaderScope::resolve`] (or [`ReaderScope::from_process`]),
//!   which always applies it, and the fields are private.
//! - A hidden id must read exactly like a missing one: callers return the same
//!   `None` / empty result for both, never a "forbidden" error.
//! - A visible memory never NAMES a hidden one: [`ReaderScope::redact_links`]
//!   drops `related_ids`, `parent_id`, `supersedes_id` and `metadata.superseded_by`
//!   entries whose target the reader cannot see (an unknown target is dropped
//!   too, so a dangling link and a hidden one read the same).
//! - Raw handles (a Cypher or SQL client, the SQLite connection, anything that
//!   returns counts or existence for the whole store) take an [`OwnerScope`],
//!   which only [`OwnerScope::from_process`] can build (public) and which refuses
//!   when a `SKMEMORY_READER` ceiling is declared in the REAL process environment.
//!   An injected `Env` is test-only (see the `testing` feature).
//!
//! # The R-08 shim must pass the audience config
//!
//! Python ships `audience_config.json` inside its package and loads it by
//! default. This crate ships no config: [`AudienceConfig::from_env`] reads
//! `SKMEMORY_AUDIENCE_CONFIG` only, and an unset variable is the EMPTY config,
//! under which every `channel:<id>` reader resolves to `@public`. That is safe
//! (it only narrows) but it silently stops `channel:` readers and `never_share`
//! tags from working. The R-08 shim that wires this crate behind the Python API
//! MUST therefore set `SKMEMORY_AUDIENCE_CONFIG` to the packaged
//! `skmemory/data/audience_config.json` (or load it with
//! [`AudienceConfig::load`]) before building any scope, and a parity test should
//! assert that a configured `channel:` reader sees more than `@public`.
//!
//! Reader grammar (the string a caller declares):
//!
//! | reader | meaning |
//! |---|---|
//! | none | the owner, unless `SKMEMORY_READER` is set |
//! | `@chef`, `@chef-only`, `owner` | the owner, no filter |
//! | `@public` `@community` `@work-circle` `@inner-circle` | that level and below |
//! | `@work:<x>`, `@inner:<x>` | the level below plus its own scoped tag |
//! | `channel:<id>` | a configured channel: members' MIN trust, its own tag, members' never-share tags |
//! | anything else | `@public` |

use crate::env::{casefold, Env};
use crate::model::py_strip;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

// `testing` exposes the env-injecting scope constructors, which can ignore the real
// process ceiling. Only skmem's own test builds may compile them: a downstream
// crate (or a wheel) enabling the feature must fail to build.
#[cfg(feature = "testing")]
const _: () = assert!(
    option_env!("CARGO_PRIMARY_PACKAGE").is_some(),
    "the `testing` feature may only be enabled by skmem's own test builds (it exposes constructors that bypass the process reader ceiling)"
);
#[cfg(all(feature = "testing", feature = "extension-module"))]
compile_error!(
    "the `testing` feature must never be combined with `extension-module` (a shipped wheel)"
);

/// The process-wide reader ceiling variable.
pub const READER_ENV: &str = "SKMEMORY_READER";
/// Where [`AudienceConfig::from_env`] looks for `audience_config.json`.
pub const AUDIENCE_CONFIG_ENV: &str = "SKMEMORY_AUDIENCE_CONFIG";

const TAG_LEVELS: [(&str, i64); 5] = [
    ("@public", 0),
    ("@community", 1),
    ("@work-circle", 2),
    ("@inner-circle", 3),
    ("@chef-only", 4),
];

const OWNER_READERS: [&str; 3] = ["@chef", "@chef-only", "owner"];

/// `audience.tag_to_level`: exact tag, then the scoped prefixes, else the most
/// restrictive level (4).
pub fn tag_to_level(tag: &str) -> i64 {
    if tag.is_empty() {
        return 4;
    }
    if let Some((_, l)) = TAG_LEVELS.iter().find(|(t, _)| *t == tag) {
        return *l;
    }
    if tag.starts_with("@work:") {
        return 2;
    }
    if tag.starts_with("@inner:") {
        return 3;
    }
    4
}

fn tags_up_to(level: i64) -> BTreeSet<String> {
    TAG_LEVELS
        .iter()
        .filter(|(_, l)| *l <= level)
        .map(|(t, _)| (*t).to_string())
        .collect()
}

/// `audience_config.json`: channels and people. Missing, unreadable or
/// malformed config is EMPTY (every channel is then unknown, hence `@public`),
/// as in Python.
#[derive(Clone, Debug, Default)]
pub struct AudienceConfig {
    config: Value,
}

impl AudienceConfig {
    /// No channels, no people.
    pub fn empty() -> AudienceConfig {
        AudienceConfig::default()
    }

    /// Parse config text; invalid JSON is the empty config.
    pub fn from_json(text: &str) -> AudienceConfig {
        match serde_json::from_str::<Value>(text) {
            Ok(v @ Value::Object(_)) => AudienceConfig { config: v },
            _ => AudienceConfig::empty(),
        }
    }

    /// Load from a file; a missing or unreadable file is the empty config.
    pub fn load(path: impl AsRef<Path>) -> AudienceConfig {
        std::fs::read_to_string(path)
            .map(|t| AudienceConfig::from_json(&t))
            .unwrap_or_default()
    }

    /// `SKMEMORY_AUDIENCE_CONFIG` when set, otherwise the empty config. Python
    /// ships a default file inside its package; this crate has none, so an
    /// unconfigured process resolves every `channel:<id>` reader to `@public`.
    pub fn from_env(env: &Env) -> AudienceConfig {
        match env.get(AUDIENCE_CONFIG_ENV).map(py_strip) {
            Some(p) if !p.is_empty() => AudienceConfig::load(p),
            _ => AudienceConfig::empty(),
        }
    }

    fn channel(&self, id: &str) -> Option<&Value> {
        self.config.get("channels")?.get(id)
    }

    fn person(&self, name: &str) -> Option<&Value> {
        self.config.get("people")?.get(name)
    }

    /// A person's trust level, failing closed. Mirrors Python
    /// `AudienceResolver.get_person_trust` (SKMEM-P0-20): an unknown person, a
    /// person with no `trust_level`, and a `trust_level` that is not an
    /// integer 0 to 4 (strings, floats, booleans, null) are all level 0
    /// (public). A level is never invented upward.
    fn trust_of(&self, name: &str) -> i64 {
        let Some(person) = self
            .person(name)
            .filter(|p| p.as_object().is_some_and(|o| !o.is_empty()))
        else {
            return 0;
        };
        match person.get("trust_level").and_then(Value::as_i64) {
            Some(l) if (0..=4).contains(&l) => l,
            _ => 0,
        }
    }
}

/// The caller is not the owner: a `SKMEMORY_READER` ceiling is declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotOwner {
    /// The declared ceiling, as written.
    pub declared: String,
}

impl std::fmt::Display for NotOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "owner access refused: {READER_ENV} declares a reader ceiling ({:?})",
            self.declared
        )
    }
}

impl std::error::Error for NotOwner {}

/// Proof that the caller is the owner. Every raw handle (a SQL or Cypher
/// client, the SQLite connection) and every store-wide count or existence
/// answer takes one, so a reader-facing code path cannot reach them.
///
/// The field is private and there is no `Default`, `Clone` or deserializer:
/// the only public constructor is [`OwnerScope::from_process`], which REFUSES when `SKMEMORY_READER`
/// declares anything but an owner reader. A process serving a group chat
/// therefore cannot obtain one by accident.
///
/// ```compile_fail
/// // the field is private: no struct literal outside this module
/// use skmem::audience::{OwnerScope, ReaderScope};
/// let r = ReaderScope::from_process(None);
/// let _o = OwnerScope { scope: r };
/// ```
///
/// ```compile_fail
/// // a reader scope is not an owner scope: the raw connection refuses it
/// use skmem::audience::ReaderScope;
/// let idx = skmem::index::Index::open("/nonexistent", true).unwrap();
/// let r = ReaderScope::from_process(None);
/// let _ = idx.connection(&r);
/// ```
#[derive(Debug)]
pub struct OwnerScope {
    scope: ReaderScope,
}

impl OwnerScope {
    /// The owner, unless `env` declares a reader ceiling that is not an owner
    /// reader (`@chef`, `@chef-only`, `owner`).
    ///
    /// Test-only (`cfg(test)` or the `testing` feature): an injected `Env` lets
    /// a caller ignore the REAL process ceiling, so production code cannot
    /// call it. The only public constructor is [`OwnerScope::from_process`].
    #[cfg(any(test, feature = "testing"))]
    pub fn resolve(env: &Env) -> Result<OwnerScope, NotOwner> {
        OwnerScope::resolve_in(env)
    }

    pub(crate) fn resolve_in(env: &Env) -> Result<OwnerScope, NotOwner> {
        let declared = env.get(READER_ENV).map(py_strip).filter(|e| !e.is_empty());
        if let Some(d) = declared {
            let scope = scope_for(d, &AudienceConfig::empty());
            if !scope.unrestricted() {
                return Err(NotOwner {
                    declared: d.to_string(),
                });
            }
        }
        Ok(OwnerScope {
            scope: ReaderScope::owner_scope(),
        })
    }

    /// [`OwnerScope::resolve`] against the live process environment.
    pub fn from_process() -> Result<OwnerScope, NotOwner> {
        OwnerScope::resolve_in(&Env::process())
    }

    /// The unrestricted read scope, for the read APIs that take a
    /// [`ReaderScope`].
    pub fn reader_scope(&self) -> &ReaderScope {
        &self.scope
    }
}

/// What a declared reader may receive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReaderScope {
    /// Exact `context_tag` values permitted; `None` is unrestricted (owner).
    allowed: Option<BTreeSet<String>>,
    /// Memory tags that must never reach this reader.
    exclusions: BTreeSet<String>,
    label: String,
}

impl ReaderScope {
    fn owner_scope() -> ReaderScope {
        ReaderScope {
            allowed: None,
            exclusions: BTreeSet::new(),
            label: "owner".into(),
        }
    }

    fn public(label: &str) -> ReaderScope {
        ReaderScope {
            allowed: Some(tags_up_to(0)),
            exclusions: BTreeSet::new(),
            label: label.into(),
        }
    }

    /// `resolve_reader`: the per-call `reader` combined with the
    /// `SKMEMORY_READER` ceiling in `env`. Both absent is the owner.
    ///
    /// Test-only (`cfg(test)` or the `testing` feature), for the same reason as
    /// [`OwnerScope::resolve`]: an injected `Env` can leave out the real
    /// process ceiling. Production code uses [`ReaderScope::from_process`] or
    /// [`ReaderScope::from_process_with_config`].
    #[cfg(any(test, feature = "testing"))]
    pub fn resolve(reader: Option<&str>, env: &Env, cfg: &AudienceConfig) -> ReaderScope {
        ReaderScope::resolve_in(reader, env, cfg)
    }

    pub(crate) fn resolve_in(reader: Option<&str>, env: &Env, cfg: &AudienceConfig) -> ReaderScope {
        ReaderScope::resolve_split(reader, env, cfg, cfg)
    }

    /// The per-call reader is resolved under `call_cfg`, the `SKMEMORY_READER`
    /// ceiling under `ceiling_cfg`, and the two are intersected. A caller that
    /// supplies `call_cfg` must never be able to widen the ceiling, so
    /// process-facing code passes a `ceiling_cfg` the caller cannot choose.
    pub(crate) fn resolve_split(
        reader: Option<&str>,
        env: &Env,
        call_cfg: &AudienceConfig,
        ceiling_cfg: &AudienceConfig,
    ) -> ReaderScope {
        let call = reader.map_or_else(ReaderScope::owner_scope, |r| scope_for(r, call_cfg));
        match env.get(READER_ENV).map(py_strip).filter(|e| !e.is_empty()) {
            Some(e) => scope_for(e, ceiling_cfg).narrow(&call),
            None => call,
        }
    }

    /// [`ReaderScope::resolve`] against the live process environment and
    /// `SKMEMORY_AUDIENCE_CONFIG`.
    pub fn from_process(reader: Option<&str>) -> ReaderScope {
        let env = Env::process();
        let cfg = AudienceConfig::from_env(&env);
        ReaderScope::resolve_in(reader, &env, &cfg)
    }

    /// As [`ReaderScope::from_process`] with an explicit audience config (the
    /// R-08 shim loads the packaged `audience_config.json` and passes it here).
    ///
    /// `cfg` resolves only the per-call `reader`. The `SKMEMORY_READER`
    /// ceiling is resolved under the config the CALLER CANNOT CHOOSE
    /// (`SKMEMORY_AUDIENCE_CONFIG` from the process environment; unset means the
    /// empty config, so a `channel:` ceiling is `@public`, fail closed), and the
    /// result is the intersection: `cfg` can only narrow, never widen.
    pub fn from_process_with_config(reader: Option<&str>, cfg: &AudienceConfig) -> ReaderScope {
        let env = Env::process();
        let ceiling_cfg = AudienceConfig::from_env(&env);
        ReaderScope::resolve_split(reader, &env, cfg, &ceiling_cfg)
    }

    /// Drop the links of `m` that name a memory this reader cannot see:
    /// `related_ids` entries, `parent_id`, `supersedes_id` and
    /// `metadata.superseded_by` (removed whole when its `id` is hidden, since
    /// its other fields would still say a hidden successor exists).
    /// `visible(id)` answers for the TARGET and must be false for an id it
    /// cannot find, so a dangling link reads like a hidden one. The owner is a
    /// no-op. Python does not do this yet (see the card notes).
    pub fn redact_links(
        &self,
        m: &mut crate::model::Memory,
        visible: &mut dyn FnMut(&str) -> bool,
    ) {
        if self.unrestricted() {
            return;
        }
        m.related_ids.retain(|i| visible(i));
        if m.parent_id.as_deref().is_some_and(|i| !visible(i)) {
            m.parent_id = None;
        }
        if m.supersedes_id.as_deref().is_some_and(|i| !visible(i)) {
            m.supersedes_id = None;
        }
        let hidden_successor = match m.metadata.get("superseded_by") {
            None => false,
            Some(v) => match v.get("id").and_then(Value::as_str) {
                Some(id) => !visible(id),
                // a link with no readable id cannot be vetted
                None => true,
            },
        };
        if hidden_successor {
            m.metadata.remove("superseded_by");
        }
    }

    /// The intersection of two scopes (most restrictive of each dimension).
    pub fn narrow(&self, other: &ReaderScope) -> ReaderScope {
        let allowed = match (&self.allowed, &other.allowed) {
            (None, o) => o.clone(),
            (s, None) => s.clone(),
            (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect()),
        };
        ReaderScope {
            allowed,
            exclusions: self.exclusions.union(&other.exclusions).cloned().collect(),
            label: format!("{}&{}", self.label, other.label),
        }
    }

    /// True for the owner with no exclusions: nothing is filtered.
    pub fn unrestricted(&self) -> bool {
        self.allowed.is_none() && self.exclusions.is_empty()
    }

    /// The reader as declared, for logs.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Allowed `context_tag` values, sorted; `None` for unrestricted.
    pub fn allowed_tags(&self) -> Option<Vec<String>> {
        self.allowed.as_ref().map(|s| s.iter().cloned().collect())
    }

    /// Never-share tags, sorted.
    pub fn exclusions(&self) -> Vec<String> {
        self.exclusions.iter().cloned().collect()
    }

    /// Whether a memory with this context tag and tags may reach the reader.
    /// A missing context tag is `""`, which no reader but the owner allows.
    pub fn allows(&self, context_tag: Option<&str>, tags: &[String]) -> bool {
        if let Some(a) = &self.allowed {
            if !a.contains(context_tag.unwrap_or("")) {
                return false;
            }
        }
        !tags.iter().any(|t| self.exclusions.contains(t))
    }

    /// [`ReaderScope::allows`] for a hydrated memory.
    pub fn allows_memory(&self, m: &crate::model::Memory) -> bool {
        self.allows(Some(&m.context_tag), &m.tags)
    }

    /// The `context_tag` filter to push into a backend, intersected with an
    /// optional caller-supplied exact tag or any-of list (Python
    /// `_search_as_reader`). `Ok(None)`: no tag restriction. `Ok(Some(vec))`:
    /// the permitted set (may be empty, which matches nothing). The caller
    /// still runs [`ReaderScope::allows`] on every row it hydrates.
    pub fn context_filter(&self, wanted: Option<&[String]>) -> Option<Vec<String>> {
        match (&self.allowed, wanted) {
            (None, None) => None,
            (None, Some(w)) => Some(w.to_vec()),
            (Some(a), None) => Some(a.iter().cloned().collect()),
            (Some(a), Some(w)) => Some(w.iter().filter(|t| a.contains(*t)).cloned().collect()),
        }
    }
}

fn scope_for(reader: &str, cfg: &AudienceConfig) -> ReaderScope {
    let name = py_strip(reader);
    let low = casefold(name);
    if OWNER_READERS.contains(&low.as_str()) {
        return ReaderScope {
            label: low,
            ..ReaderScope::owner_scope()
        };
    }
    if let Some((_, l)) = TAG_LEVELS.iter().find(|(t, _)| *t == low) {
        return ReaderScope {
            allowed: Some(tags_up_to(*l)),
            exclusions: BTreeSet::new(),
            label: low,
        };
    }
    for (prefix, base) in [("@work:", 2_i64), ("@inner:", 3_i64)] {
        if low.starts_with(prefix) && low.len() > prefix.len() {
            let mut allowed = tags_up_to(base - 1);
            allowed.insert(low.clone());
            return ReaderScope {
                allowed: Some(allowed),
                exclusions: BTreeSet::new(),
                label: low,
            };
        }
    }
    if low.starts_with("channel:") && low.len() > "channel:".len() {
        // channel ids are case-sensitive in the config: use the declared spelling
        let channel_id = name.split_once(':').map_or("", |(_, id)| id);
        return channel_scope(channel_id, &low, cfg);
    }
    ReaderScope::public("@public(unparsed)")
}

fn channel_scope(channel_id: &str, label: &str, cfg: &AudienceConfig) -> ReaderScope {
    let Some(chan) = cfg.channel(channel_id) else {
        return ReaderScope::public(label);
    };
    let members: Vec<&str> = chan
        .get("members")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if members.is_empty() {
        return ReaderScope::public(label);
    }
    let min_trust = members.iter().map(|m| cfg.trust_of(m)).min().unwrap_or(0);
    let mut exclusions = BTreeSet::new();
    for m in &members {
        if let Some(list) = cfg
            .person(m)
            .and_then(|p| p.get("never_share"))
            .and_then(Value::as_array)
        {
            exclusions.extend(list.iter().filter_map(Value::as_str).map(str::to_string));
        }
    }
    let context_tag = chan
        .get("context_tag")
        .and_then(Value::as_str)
        .unwrap_or("@chef-only");
    let mut allowed = tags_up_to(min_trust);
    if !context_tag.is_empty() && tag_to_level(context_tag) <= min_trust {
        allowed.insert(context_tag.to_string());
    }
    ReaderScope {
        allowed: Some(allowed),
        exclusions,
        label: label.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(reader: Option<&str>, env: &Env) -> ReaderScope {
        ReaderScope::resolve_in(reader, env, &AudienceConfig::empty())
    }

    #[test]
    fn owner_when_nothing_declared() {
        let sc = s(None, &Env::empty());
        assert!(sc.unrestricted());
        assert!(sc.allows(Some("@chef-only"), &[]));
        assert!(sc.allows(None, &[]));
    }

    #[test]
    fn env_is_a_ceiling_per_call_cannot_widen() {
        let env = Env::empty().with(READER_ENV, "@public");
        for wide in [
            "@chef-only",
            "owner",
            "@inner-circle",
            "@work:x",
            "channel:x",
        ] {
            let sc = s(Some(wide), &env);
            assert!(!sc.allows(Some("@work-circle"), &[]), "{wide} widened");
            assert!(!sc.allows(Some("@chef-only"), &[]), "{wide} widened");
            assert!(sc.allows(Some("@public"), &[]));
        }
        let none = s(None, &env);
        assert!(!none.allows(Some("@community"), &[]));
    }

    #[test]
    fn per_call_narrows_env() {
        let env = Env::empty().with(READER_ENV, "@inner-circle");
        let sc = s(Some("@community"), &env);
        assert!(sc.allows(Some("@community"), &[]));
        assert!(!sc.allows(Some("@work-circle"), &[]));
    }

    #[test]
    fn unknown_reader_is_public_never_wider() {
        for r in [
            "",
            "   ",
            "root",
            "admin",
            "@everyone",
            "@work:",
            "channel:",
            "*",
        ] {
            let sc = s(Some(r), &Env::empty());
            assert!(sc.allows(Some("@public"), &[]), "{r:?}");
            assert!(!sc.allows(Some("@community"), &[]), "{r:?}");
            assert!(!sc.allows(None, &[]), "{r:?}");
        }
    }

    #[test]
    fn blank_env_is_not_a_declaration() {
        let sc = s(None, &Env::empty().with(READER_ENV, "   "));
        assert!(sc.unrestricted());
    }

    #[test]
    fn scoped_tags_isolate_clients() {
        let sc = s(Some("@work:a"), &Env::empty());
        assert!(sc.allows(Some("@work:a"), &[]));
        assert!(!sc.allows(Some("@work:b"), &[]));
        assert!(sc.allows(Some("@community"), &[]));
        assert!(!sc.allows(Some("@work-circle"), &[]));
    }

    #[test]
    fn missing_context_tag_is_hidden_from_non_owner() {
        assert!(!s(Some("@inner-circle"), &Env::empty()).allows(None, &[]));
        assert!(!s(Some("@inner-circle"), &Env::empty()).allows(Some(""), &[]));
    }

    #[test]
    fn channel_resolution_and_exclusions() {
        let cfg = AudienceConfig::from_json(
            r#"{"channels":{"c1":{"context_tag":"@work:x","members":["A","B"]},
                "empty":{"context_tag":"@chef-only","members":[]}},
               "people":{"A":{"trust_level":2,"never_share":["romantic"]},
                         "B":{"trust_level":3,"never_share":["revenue"]}}}"#,
        );
        let sc = ReaderScope::resolve_in(Some("channel:c1"), &Env::empty(), &cfg);
        assert!(sc.allows(Some("@work:x"), &[]));
        assert!(sc.allows(Some("@work-circle"), &[]));
        assert!(!sc.allows(Some("@inner-circle"), &[]));
        assert!(!sc.allows(Some("@work-circle"), &["revenue".into()]));
        assert!(!sc.allows(Some("@work-circle"), &["romantic".into()]));
        let e = ReaderScope::resolve_in(Some("channel:empty"), &Env::empty(), &cfg);
        assert_eq!(e.allowed_tags().unwrap(), ["@public"]);
        let u = ReaderScope::resolve_in(Some("channel:nope"), &Env::empty(), &cfg);
        assert_eq!(u.allowed_tags().unwrap(), ["@public"]);
    }

    #[test]
    fn malformed_trust_level_only_narrows() {
        let cfg = AudienceConfig::from_json(
            r#"{"channels":{"c":{"context_tag":"@chef-only","members":["A"]}},
                "people":{"A":{"trust_level":"4"}}}"#,
        );
        let sc = ReaderScope::resolve_in(Some("channel:c"), &Env::empty(), &cfg);
        assert_eq!(sc.allowed_tags().unwrap(), ["@public"]);
    }

    #[test]
    fn unknown_member_and_missing_level_fail_closed() {
        for people in [
            r#"{}"#,
            r#"{"A":{"never_share":[]}}"#,
            r#"{"A":{"trust_level":null}}"#,
            r#"{"A":{"trust_level":2.0}}"#,
            r#"{"A":{"trust_level":true}}"#,
            r#"{"A":{"trust_level":9}}"#,
        ] {
            let cfg = AudienceConfig::from_json(&format!(
                r#"{{"channels":{{"c":{{"context_tag":"@chef-only","members":["A"]}}}},"people":{people}}}"#
            ));
            let sc = ReaderScope::resolve(Some("channel:c"), &Env::empty(), &cfg);
            assert_eq!(sc.allowed_tags().unwrap(), ["@public"], "{people}");
        }
        // one unknown member drags a trusted channel to public
        let cfg = AudienceConfig::from_json(
            r#"{"channels":{"c":{"members":["A","Ghost"]}},"people":{"A":{"trust_level":4}}}"#,
        );
        let sc = ReaderScope::resolve(Some("channel:c"), &Env::empty(), &cfg);
        assert_eq!(sc.allowed_tags().unwrap(), ["@public"]);
    }

    #[test]
    fn bad_config_text_is_empty() {
        let cfg = AudienceConfig::from_json("not json");
        let sc = ReaderScope::resolve_in(Some("channel:c1"), &Env::empty(), &cfg);
        assert_eq!(sc.allowed_tags().unwrap(), ["@public"]);
    }

    #[test]
    fn context_filter_intersects() {
        let sc = s(Some("@community"), &Env::empty());
        let want = vec!["@public".to_string(), "@chef-only".to_string()];
        assert_eq!(sc.context_filter(Some(&want)).unwrap(), ["@public"]);
        assert_eq!(
            sc.context_filter(Some(&["@chef-only".into()]))
                .unwrap()
                .len(),
            0
        );
        assert!(s(None, &Env::empty()).context_filter(None).is_none());
    }
}
