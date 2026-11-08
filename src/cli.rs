//! The `skmem` command line: the hot path of the Python `skmemory` CLI on the
//! Rust engine (SKMEM-R-10).
//!
//! Commands: `search`, `recall` (alias `show`), `list`, `health`, `snapshot`,
//! `reindex`, `intentions {list,done}`, `ledger top`. Flags and plain-text output
//! match `skmemory/cli.py`; `recall` and `health` print JSON like Python does.
//! `--json` on `search`, `list` and `intentions list` is an addition (Python has
//! no machine output for them): the rows are the `Memory.model_dump()` dicts the
//! PyO3 `Engine.search` / `Engine.list` return, plus `superseded` and
//! `superseded_by` on search.
//!
//! # Scope
//!
//! `SKMEMORY_READER` is a ceiling, resolved by the engine on every call
//! ([`ReaderScope::from_process_with_config`]); `--reader` can only narrow it.
//! A declared reader opens the store read-only, never reinforces, gets
//! [`redact_links`] on `recall`, and is refused every owner-only command
//! (`snapshot`, `reindex`, `intentions`, `ledger`). Python allows `ledger top`
//! to a reader; here it is owner-only, which is stricter, never wider.
//!
//! # Store location
//!
//! `--base DIR` (the memory directory itself), else `SKMEMORY_HOME/memory`
//! (`SKMEMORY_HOME` is the agent directory, as in Python's `config.py`), else
//! `$SKCAPSTONE_HOME|~/.skcapstone` `/agents/<agent>/memory` for the agent the
//! selectors name. `SKAGENT`, `SKCAPSTONE_AGENT` and `SKMEMORY_AGENT` must
//! agree; with none set, `SK_DEFAULT_AGENT`, then a sole installed agent;
//! otherwise it refuses rather than guessing. (The Python profile-registry
//! health check is not repeated here.)

use crate::audience::{OwnerScope, ReaderScope};
use crate::engine::{Arms, Engine, EngineError, EngineOptions, SearchReport, SearchRequest};
use crate::env::Env;
use crate::index::{wal_status, WAL_DEGRADED_BYTES};
use crate::intentions::IntentionStore;
use crate::ledger::Ledger;
use crate::model::{Layer, Memory};
use crate::pyjson::dumps_indent2;
use crate::pytime::Dt;
use crate::snapshot::SnapshotError;
use crate::validity::normalize_as_of;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

const OWNER_ONLY: &str = "owner-only: unavailable while SKMEMORY_READER declares a non-owner.";
/// Python's `SEARCH_HIT_DEBOUNCE_S`.
const SEARCH_HIT_DEBOUNCE_S: f64 = 3600.0;

/// A command failure: the message for stderr and the exit status.
struct Fail {
    code: i32,
    msg: String,
    /// Click prints `Error: ...` for usage errors and ClickExceptions, but the
    /// bare message for `click.echo(..., err=True)` plus `sys.exit(1)`.
    bare: bool,
}

type Res<T> = Result<T, Fail>;

fn usage(msg: impl Into<String>) -> Fail {
    Fail {
        code: 2,
        msg: msg.into(),
        bare: false,
    }
}

fn click_err(msg: impl Into<String>) -> Fail {
    Fail {
        code: 1,
        msg: msg.into(),
        bare: false,
    }
}

fn bare(msg: impl Into<String>) -> Fail {
    Fail {
        code: 1,
        msg: msg.into(),
        bare: true,
    }
}

fn engine_fail(e: EngineError) -> Fail {
    click_err(e.to_string())
}

// ---------------------------------------------------------------------------
// argument parsing (Click-shaped: `--opt value`, `--opt=value`, flags)
// ---------------------------------------------------------------------------

struct Spec {
    /// Options that take a value.
    value: &'static [&'static str],
    /// Boolean flags.
    flag: &'static [&'static str],
    /// How many positionals (name list for the error message).
    positional: &'static [&'static str],
}

struct Parsed {
    values: HashMap<&'static str, String>,
    flags: BTreeSet<&'static str>,
    pos: Vec<String>,
}

impl Parsed {
    fn get(&self, k: &str) -> Option<&str> {
        self.values.get(k).map(String::as_str)
    }
    fn has(&self, k: &str) -> bool {
        self.flags.contains(k)
    }
    fn int(&self, k: &'static str, default: i64) -> Res<i64> {
        match self.get(k) {
            None => Ok(default),
            Some(s) => s.trim().parse::<i64>().map_err(|_| {
                usage(format!(
                    "Invalid value for '--{k}': '{s}' is not a valid integer."
                ))
            }),
        }
    }
    fn float(&self, k: &'static str, default: f64) -> Res<f64> {
        match self.get(k) {
            None => Ok(default),
            Some(s) => s.trim().parse::<f64>().map_err(|_| {
                usage(format!(
                    "Invalid value for '--{k}': '{s}' is not a valid float."
                ))
            }),
        }
    }
}

fn parse(spec: &Spec, args: &[String]) -> Res<Parsed> {
    let mut p = Parsed {
        values: HashMap::new(),
        flags: BTreeSet::new(),
        pos: Vec::new(),
    };
    let mut i = 0;
    let mut only_pos = false;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if only_pos || !a.starts_with("--") || a == "--" {
            if a == "--" && !only_pos {
                only_pos = true;
            } else {
                p.pos.push(a.clone());
            }
            continue;
        }
        let body = &a[2..];
        let (name, inline) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (body, None),
        };
        if let Some(&k) = spec.value.iter().find(|k| **k == name) {
            let v = match inline {
                Some(v) => v,
                None => {
                    let v = args
                        .get(i)
                        .cloned()
                        .ok_or_else(|| usage(format!("Option '--{name}' requires an argument.")))?;
                    i += 1;
                    v
                }
            };
            p.values.insert(k, v);
        } else if let Some(&k) = spec.flag.iter().find(|k| **k == name) {
            if inline.is_some() {
                return Err(usage(format!("Option '--{name}' does not take a value.")));
            }
            p.flags.insert(k);
        } else if let Some(neg) = name.strip_prefix("no-") {
            // Click's `--x/--no-x` pairs: `--no-x` clears `x`.
            match spec.flag.iter().find(|k| **k == neg) {
                Some(&k) => {
                    p.flags.remove(k);
                }
                None => return Err(usage(format!("No such option: --{name}"))),
            }
        } else {
            return Err(usage(format!("No such option: --{name}")));
        }
    }
    if p.pos.len() < spec.positional.len() {
        let missing = spec.positional[p.pos.len()].to_uppercase();
        return Err(usage(format!("Missing argument '{missing}'.")));
    }
    if p.pos.len() > spec.positional.len() {
        return Err(usage(format!(
            "Got unexpected extra argument ({})",
            p.pos[spec.positional.len()..].join(" ")
        )));
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// store location and engine
// ---------------------------------------------------------------------------

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn agents_base() -> PathBuf {
    if let Some(h) = env_nonempty("SKCAPSTONE_HOME") {
        return PathBuf::from(h).join("agents");
    }
    let home = env_nonempty("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".skcapstone").join("agents")
}

/// The agent the selectors name, refusing to guess (`agents.get_active_agent`).
fn active_agent(base: &Path) -> Result<String, String> {
    let mut chosen: Vec<String> = Vec::new();
    for v in ["SKAGENT", "SKCAPSTONE_AGENT", "SKMEMORY_AGENT"] {
        if let Some(a) = env_nonempty(v) {
            chosen.push(a);
        }
    }
    chosen.dedup();
    let mut distinct = chosen.clone();
    distinct.sort();
    distinct.dedup();
    if distinct.len() > 1 {
        return Err(format!(
            "conflicting agent selectors ({}); set all of SKAGENT, SKCAPSTONE_AGENT and \
             SKMEMORY_AGENT to one name or none",
            distinct.join(", ")
        ));
    }
    if let Some(a) = distinct.pop() {
        return Ok(a);
    }
    let mut installed: Vec<String> = std::fs::read_dir(base)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| e.path().join("memory").is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !n.ends_with("-template") && !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    installed.sort();
    if let Some(d) = env_nonempty("SK_DEFAULT_AGENT") {
        return if installed.contains(&d) {
            Ok(d)
        } else {
            Err(format!("SK_DEFAULT_AGENT={d} is not an installed agent"))
        };
    }
    if installed.len() == 1 {
        return Ok(installed.remove(0));
    }
    Err(format!(
        "no active agent could be resolved without guessing; set SKAGENT (installed: {})",
        if installed.is_empty() {
            "none".to_string()
        } else {
            installed.join(", ")
        }
    ))
}

fn memory_dir(base_opt: Option<&str>) -> Res<PathBuf> {
    if let Some(b) = base_opt.filter(|b| !b.is_empty()) {
        return Ok(PathBuf::from(b));
    }
    if let Some(h) = env_nonempty("SKMEMORY_HOME") {
        return Ok(PathBuf::from(h).join("memory"));
    }
    let base = agents_base();
    let agent = active_agent(&base).map_err(|e| bare(format!("No active agent memory profile: {e}")))?;
    Ok(base.join(agent).join("memory"))
}

fn agent_of(memory_dir: &Path) -> Option<String> {
    if memory_dir.file_name()?.to_str()? != "memory" {
        return None;
    }
    let agent = memory_dir.parent()?;
    if agent.parent()?.file_name()?.to_str()? != "agents" {
        return None;
    }
    agent.file_name()?.to_str().map(str::to_string)
}

#[cfg(feature = "pg")]
fn build_arms(agent: Option<&str>, pg_read_only: bool) -> Arms {
    use crate::engine::pg_arms::{attach, PgConfig};
    use crate::engine::Attach;
    let Some(cfg) = PgConfig::from_env(pg_read_only) else {
        return Arms::none();
    };
    match agent {
        Some(a) => attach(&cfg, a),
        None => {
            let why = "agent unknown: open <home>/agents/<agent>/memory";
            Arms {
                semantic: Attach::Failed(why.into()),
                lexical: Attach::Failed(why.into()),
                graph: Attach::Absent,
            }
        }
    }
}

#[cfg(not(feature = "pg"))]
fn build_arms(_agent: Option<&str>, _pg_read_only: bool) -> Arms {
    Arms::none()
}

/// A `--no-vector` run leaves the Postgres arms off, like Python's.
fn open_engine(dir: &Path, no_vector: bool) -> Res<Engine> {
    let is_owner = OwnerScope::from_process().is_ok();
    let arms = if no_vector {
        Arms::none()
    } else {
        build_arms(agent_of(dir).as_deref(), !is_owner)
    };
    let audience_config = env_nonempty("SKMEMORY_AUDIENCE_CONFIG").map(PathBuf::from);
    Engine::open(
        dir,
        EngineOptions {
            audience_config,
            fts: true,
            read_only: Some(!is_owner),
            arms,
        },
    )
    .map_err(engine_fail)
}

fn owner() -> Res<OwnerScope> {
    OwnerScope::from_process().map_err(|_| bare(OWNER_ONLY))
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn chars_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn csv(s: Option<&str>) -> Vec<String> {
    s.unwrap_or("")
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn parse_layer_opt(v: Option<&str>) -> Res<Option<Layer>> {
    match v {
        None => Ok(None),
        Some(s) => Layer::parse(s).map(Some).ok_or_else(|| {
            usage(format!(
                "Invalid value for '--layer': '{s}' is not one of 'short-term', 'mid-term', \
                 'long-term'."
            ))
        }),
    }
}

/// `validity.superseded_by_map`, restricted to ids the reader may see.
fn superseded_map(
    mems: &[Memory],
    visible: &mut dyn FnMut(&str) -> bool,
) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    for m in mems {
        if let Some(s) = m.supersedes_id.as_deref().filter(|s| !s.is_empty()) {
            out.insert(s.to_string(), m.id.clone());
        }
    }
    for m in mems {
        if let Some(sid) = m
            .metadata
            .get("superseded_by")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            out.entry(m.id.clone()).or_insert_with(|| sid.to_string());
        }
    }
    let by_id: HashMap<&str, &Memory> = mems.iter().map(|m| (m.id.as_str(), m)).collect();
    out.into_iter()
        .filter(|(k, _)| by_id.get(k.as_str()).is_some_and(|m| m.valid_to.is_some()))
        .filter(|(_, v)| visible(v))
        .collect()
}

fn visible(eng: &Engine, id: &str, scope: &ReaderScope) -> bool {
    eng.store().load(id, scope).ok().flatten().is_some()
}

// ---------------------------------------------------------------------------
// redact_links (Python MemoryStore.redact_links)
// ---------------------------------------------------------------------------

/// `observations.redact_observation_text`: drop quote lines whose memory is
/// hidden and restate the count. Fails closed to the bare statement.
fn redact_observation_text(
    title: &str,
    content: &str,
    evidence: &[String],
    visible: &BTreeSet<String>,
) -> (String, String) {
    let hidden: Vec<&String> = evidence.iter().filter(|i| !visible.contains(*i)).collect();
    if hidden.is_empty() {
        return (title.to_string(), content.to_string());
    }
    let n = evidence.iter().filter(|i| visible.contains(*i)).count();
    let fix = |text: &str| fix_count(text, n);
    let (head, tail) = match content.split_once("\n\n") {
        Some((h, t)) => (h, t),
        None => (content, ""),
    };
    let kept: Vec<&str> = quote_lines(tail)
        .into_iter()
        .filter(|(_, id)| visible.contains(*id))
        .map(|(whole, _)| whole)
        .collect();
    let mut new = if kept.is_empty() {
        fix(head)
    } else {
        let mut parts = vec![head, ""];
        parts.extend(kept);
        fix(&parts.join("\n"))
    };
    if hidden.iter().any(|h| new.contains(h.as_str())) {
        new = fix(head);
    }
    if hidden.iter().any(|h| new.contains(h.as_str())) {
        new = "Observation".to_string();
    }
    let mut new_title = fix(title);
    if hidden.iter().any(|h| new_title.contains(h.as_str())) {
        new_title = "Observation".to_string();
    }
    (new_title, new)
}

/// `re.sub(r"connects \d+ memories", f"connects {n} memories", text)`.
fn fix_count(text: &str, n: usize) -> String {
    const HEAD: &str = "connects ";
    const TAIL: &str = " memories";
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find(HEAD) {
        let after = &rest[at + HEAD.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after[digits..].starts_with(TAIL) {
            out.push_str(&rest[..at]);
            out.push_str(&format!("connects {n} memories"));
            rest = &after[digits + TAIL.len()..];
        } else {
            out.push_str(&rest[..at + HEAD.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// `_QUOTE_RE.finditer`: `"(.*?)" \(memory (\S+?)\)(?=\n|$)`, as `(whole, id)`.
fn quote_lines(tail: &str) -> Vec<(&str, &str)> {
    const MID: &str = "\" (memory ";
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = tail[pos..].find('"') {
        let start = pos + rel;
        let body = start + 1;
        // Shortest quote body, then shortest id, that ends at a line end.
        let mut found: Option<(usize, usize, usize)> = None;
        let mut from = body;
        while let Some(r) = tail[from..].find(MID) {
            let q_end = from + r;
            let id_start = q_end + MID.len();
            let mut j = id_start;
            // \S+? then `)` then lookahead \n or end.
            let mut hit = None;
            for (off, c) in tail[id_start..].char_indices() {
                if c.is_whitespace() {
                    break;
                }
                if off > 0 && c == ')' {
                    let after = id_start + off + 1;
                    if after == tail.len() || tail[after..].starts_with('\n') {
                        hit = Some((id_start + off, after));
                        break;
                    }
                }
                j = id_start + off + c.len_utf8();
            }
            let _ = j;
            if let Some((id_end, after)) = hit {
                found = Some((id_start, id_end, after));
                break;
            }
            from = q_end + 1;
        }
        match found {
            Some((id_start, id_end, after)) => {
                out.push((&tail[start..after], &tail[id_start..id_end]));
                pos = after;
            }
            None => pos = body,
        }
    }
    out
}

fn is_str_visible(v: &Value, vis: &mut dyn FnMut(&str) -> bool) -> bool {
    match v {
        Value::String(s) if !s.is_empty() => vis(s),
        _ => false,
    }
}

/// A copy of `m` with links to memories the reader cannot see removed.
pub fn redact_links(m: &Memory, vis: &mut dyn FnMut(&str) -> bool) -> Memory {
    let mut out = m.clone();
    let mut meta: Map<String, Value> = m.metadata.clone();
    let parent = m
        .parent_id
        .clone()
        .filter(|p| !p.is_empty())
        .filter(|p| vis(p));
    if m.parent_id.as_deref().is_some_and(|p| !p.is_empty()) && parent.is_none() {
        meta.shift_remove("parent_relation");
    }
    if let Some(sb) = meta.get("superseded_by") {
        if !sb.is_null() {
            let keep = sb
                .as_object()
                .and_then(|o| o.get("id"))
                .is_some_and(|id| is_str_visible(id, vis));
            if !keep {
                meta.shift_remove("superseded_by");
            }
        }
    }
    if let Some(Value::Array(items)) = meta.get("possible_conflict_with").cloned() {
        let kept: Vec<Value> = items.into_iter().filter(|i| is_str_visible(i, vis)).collect();
        if kept.is_empty() {
            meta.shift_remove("possible_conflict_with");
        } else {
            meta.insert("possible_conflict_with".into(), Value::Array(kept));
        }
    }
    if let Some(pid) = meta.get("promoted_id") {
        if !pid.is_null() && !is_str_visible(pid, vis) {
            meta.shift_remove("promoted_id");
        }
    }
    if let Some(Value::Array(items)) = meta.get("chunk_memory_ids").cloned() {
        let kept: Vec<Value> = items.into_iter().filter(|i| is_str_visible(i, vis)).collect();
        meta.insert("chunk_memory_ids".into(), Value::Array(kept));
    }
    let mut title = m.title.clone();
    let mut content = m.content.clone();
    if let Some(Value::Object(obs)) = meta.get("observation").cloned() {
        if let Some(Value::Array(ev)) = obs.get("evidence_ids") {
            let kept: Vec<Value> = ev.iter().filter(|i| is_str_visible(i, vis)).cloned().collect();
            let mut new_obs = obs.clone();
            new_obs.insert("evidence_ids".into(), Value::Array(kept.clone()));
            if obs.contains_key("proof_count") {
                new_obs.insert("proof_count".into(), Value::from(kept.len()));
            }
            if kept.len() != ev.len() {
                let all: Vec<String> = ev
                    .iter()
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                    .collect();
                let vis_set: BTreeSet<String> = kept
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                let (t, c) = redact_observation_text(&m.title, &m.content, &all, &vis_set);
                title = t;
                content = c;
            }
            meta.insert("observation".into(), Value::Object(new_obs));
        }
    }
    if let Some(Value::Object(wt)) = meta.get("write_triage") {
        let target_ok = wt.get("target").is_some_and(|t| is_str_visible(t, vis));
        if !target_ok {
            meta.shift_remove("write_triage");
        }
    }
    if let Some(Value::Object(dec)) = meta.get("decomposition").cloned() {
        let pid = dec.get("parent_id");
        let truthy = pid.is_some_and(|p| match p {
            Value::Null => false,
            Value::String(s) => !s.is_empty(),
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64() != Some(0.0),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        });
        if truthy && !pid.is_some_and(|p| is_str_visible(p, vis)) {
            let mut d = dec.clone();
            d.shift_remove("parent_id");
            meta.insert("decomposition".into(), Value::Object(d));
        }
    }
    out.title = title;
    out.content = content;
    out.related_ids = m
        .related_ids
        .iter()
        .filter(|i| !i.is_empty() && vis(i))
        .cloned()
        .collect();
    out.parent_id = parent;
    out.supersedes_id = m
        .supersedes_id
        .clone()
        .filter(|s| !s.is_empty())
        .filter(|s| vis(s));
    out.metadata = meta;
    out
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

const GLOBAL_VALUE: &[&str] = &["base"];
const GLOBAL_FLAG: &[&str] = &["no-vector", "ai"];

fn search_cmd(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &["limit", "reader", "as-of", "layer", "tags", "source", "base"],
        flag: &["json", "no-vector"],
        positional: &["query"],
    };
    let p = parse(&spec, args)?;
    let limit = p.int("limit", 10)?.max(0) as usize;
    let layer = parse_layer_opt(p.get("layer"))?;
    let as_of = match p.get("as-of") {
        Some(a) => normalize_as_of(a).map_err(|e| {
            usage(format!("Invalid value for '--as-of': {e}"))
        })?,
        None => None,
    };
    let tags = csv(p.get("tags"));
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, p.has("no-vector"))?;
    let reader = p.get("reader").map(str::to_string);
    let scope = eng.scope(reader.as_deref());
    let output = eng
        .search(
            &SearchRequest {
                query: &p.pos[0],
                limit,
                layer: layer.map(Layer::as_str),
                tags: &tags,
                source: p.get("source"),
                context_tags: None,
                as_of: as_of.as_deref(),
                now: None,
            },
            &scope,
            &Env::process(),
        )
        .map_err(engine_fail)?;
    let report: SearchReport = output.report;
    let mems: Vec<Memory> = output.memories.into_iter().take(limit).collect();
    if scope.unrestricted() && !eng.is_read_only() && OwnerScope::from_process().is_ok() {
        let ids: Vec<&str> = mems.iter().map(|m| m.id.as_str()).collect();
        eng.index()
            .bump_access(&ids, None, Some(SEARCH_HIT_DEBOUNCE_S), &scope);
    }
    let degraded = report.degraded;
    if let Some(d) = degraded {
        let _ = writeln!(err, "degraded: {d}");
    }
    if p.has("json") {
        let sup = superseded_map(&mems, &mut |id| visible(&eng, id, &scope));
        let rows: Vec<Value> = mems
            .iter()
            .map(|m| {
                let mut v = m.to_value();
                let o = v.as_object_mut().expect("memory is an object");
                o.insert("superseded".into(), Value::Bool(sup.contains_key(&m.id)));
                o.insert(
                    "superseded_by".into(),
                    sup.get(&m.id).map_or(Value::Null, |s| Value::String(s.clone())),
                );
                v
            })
            .collect();
        let _ = writeln!(out, "{}", dumps_indent2(&Value::Array(rows)));
        return Ok(0);
    }
    if mems.is_empty() {
        let _ = writeln!(
            out,
            "{}",
            if degraded.is_none() {
                "No memories found."
            } else {
                "No memories found (degraded)."
            }
        );
        return Ok(0);
    }
    for m in &mems {
        let _ = writeln!(
            out,
            "[{}] {}.. | {} | {}",
            m.layer.as_str(),
            chars_prefix(&m.id, 8),
            m.title,
            m.emotional.signature()
        );
        if !m.summary.is_empty() {
            let _ = writeln!(out, "  Summary: {}", chars_prefix(&m.summary, 100));
        }
        let _ = writeln!(out);
    }
    Ok(0)
}

/// Find a memory by exact id, else (id of 6+ chars) by file-name prefix across
/// the tier directories, like the Python `recall`.
fn find_for_recall(eng: &Engine, id: &str, scope: &ReaderScope) -> Res<Option<Memory>> {
    let load = |i: &str| eng.store().load(i, scope).map_err(|e| click_err(e.to_string()));
    if let Some(m) = load(id)? {
        return Ok(Some(m));
    }
    if id.chars().count() < 6 || id.contains(['/', '\\']) {
        return Ok(None);
    }
    for layer in Layer::ALL {
        let Ok(rd) = std::fs::read_dir(eng.root().join(layer.as_str())) else {
            continue;
        };
        let mut stems: Vec<String> = rd
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|n| n.strip_suffix(".json").map(str::to_string))
            .filter(|s| s.starts_with(id))
            .collect();
        stems.sort();
        for s in stems {
            if let Some(m) = load(&s)? {
                return Ok(Some(m));
            }
        }
    }
    Ok(None)
}

fn recall_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &["base"],
        flag: &["no-vector"],
        positional: &["memory_id"],
    };
    let p = parse(&spec, args)?;
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, p.has("no-vector"))?;
    let scope = eng.scope(None);
    let Some(mut m) = find_for_recall(&eng, &p.pos[0], &scope)? else {
        return Err(bare(format!("Memory not found: {}", p.pos[0])));
    };
    if !m.integrity_hash.is_empty() && !m.verify_integrity() {
        m.metadata.insert(
            "integrity_warning".into(),
            Value::String(format!(
                "Integrity check failed at {}. This memory may have been tampered with.",
                Dt::now().isoformat()
            )),
        );
    }
    if scope.unrestricted() && !eng.is_read_only() && OwnerScope::from_process().is_ok() {
        eng.index().bump_access(&[m.id.as_str()], None, None, &scope);
    }
    let m = if scope.unrestricted() {
        m
    } else {
        redact_links(&m, &mut |id| visible(&eng, id, &scope))
    };
    let _ = writeln!(out, "{}", dumps_indent2(&m.to_value()));
    Ok(0)
}

fn list_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &["layer", "tags", "limit", "source", "base"],
        flag: &["json", "no-vector"],
        positional: &[],
    };
    let p = parse(&spec, args)?;
    let limit = p.int("limit", 20)?.max(0) as usize;
    let layer = parse_layer_opt(p.get("layer"))?;
    let tags = csv(p.get("tags"));
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, p.has("no-vector"))?;
    let scope = eng.scope(None);
    let source = p.get("source");
    let listing = eng.store().list(layer, &scope);
    let rows: Vec<&Memory> = listing
        .memories
        .iter()
        .filter(|m| tags.iter().all(|t| m.tags.contains(t)))
        .filter(|m| source.is_none_or(|s| s.is_empty() || m.source == s))
        .take(limit)
        .collect();
    if p.has("json") {
        let v = Value::Array(rows.iter().map(|m| m.to_value()).collect());
        let _ = writeln!(out, "{}", dumps_indent2(&v));
        return Ok(0);
    }
    if rows.is_empty() {
        let _ = writeln!(out, "No memories found.");
        return Ok(0);
    }
    let _ = writeln!(out, "Found {} memories:\n", rows.len());
    for m in rows {
        let tag_str = if m.tags.is_empty() {
            "none".to_string()
        } else {
            m.tags.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
        };
        let _ = writeln!(
            out,
            "  [{}] {}.. | {}",
            m.layer.as_str(),
            chars_prefix(&m.id, 12),
            m.title
        );
        let _ = writeln!(out, "    Tags: {} | Emotion: {}", tag_str, m.emotional.signature());
        let _ = writeln!(out);
    }
    Ok(0)
}

fn health_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &["base"],
        flag: &["no-vector"],
        positional: &[],
    };
    let p = parse(&spec, args)?;
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, p.has("no-vector"))?;
    let scope = eng.scope(None);
    let report = health_report(&eng, &scope)?;
    let _ = writeln!(out, "{}", dumps_indent2(&report));
    Ok(0)
}

/// The engine health report (`Engine.health` of the PyO3 bindings). The owner
/// gets the report; a declared reader only the verdict.
fn health_report(eng: &Engine, scope: &ReaderScope) -> Res<Value> {
    let idx = eng.index();
    let fts_ok = !idx.fts_enabled() || idx.fts_ok();
    let own = OwnerScope::from_process().ok().filter(|_| scope.unrestricted());
    let Some(own) = own else {
        return Ok(json!({
            "engine": "skmem-rust",
            "ok": fts_ok,
            "status": if fts_ok { "ok" } else { "degraded" },
        }));
    };
    let rs = own.reader_scope();
    let flat_ids = eng.store().flat_id_set(rs);
    let sqlite_ids: BTreeSet<String> = idx
        .all_rows(rs)
        .map_err(|e| click_err(e.to_string()))?
        .into_iter()
        .map(|r| r.id)
        .collect();
    let sqlite_only = sqlite_ids.difference(&flat_ids).count();
    let flat_only = flat_ids.difference(&sqlite_ids).count();
    let unloadable = eng.store().list(None, rs).unloadable.len();
    let wal = wal_status(idx.path(), WAL_DEGRADED_BYTES);
    let fts = idx.fts_status(&own).map_err(|e| click_err(e.to_string()))?;
    let in_sync = sqlite_only == 0 && flat_only == 0;
    let ok = unloadable == 0 && wal.wal_ok && fts_ok;
    Ok(json!({
        "engine": "skmem-rust",
        "ok": ok,
        "status": if in_sync && ok { "ok" } else { "degraded" },
        "sqlite_only": sqlite_only,
        "flat_only": flat_only,
        "primary": {
            "ok": ok,
            "backend": "SkmemRust",
            "base_path": eng.root().to_string_lossy(),
            "total_memories": sqlite_ids.len(),
            "wal": {
                "wal_size_bytes": wal.wal_size_bytes,
                "wal_degraded_threshold_bytes": wal.wal_degraded_threshold_bytes,
                "wal_ok": wal.wal_ok,
            },
            "fts": fts.map(|c| json!({"total": c.total, "indexed": c.indexed, "missing": c.missing})),
            "sync": {
                "in_sync": in_sync,
                "sqlite_only": sqlite_only,
                "flat_only": flat_only,
                "unloadable": unloadable,
            },
        },
    }))
}

fn snapshot_cmd(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &[
            "layer", "role", "tags", "intensity", "valence", "emotions", "resonance", "source",
            "base",
        ],
        flag: &["decompose", "no-vector"],
        positional: &["title", "content"],
    };
    let mut p = parse(&spec, args)?;
    let layer = p.get("layer").unwrap_or("short-term").to_string();
    if Layer::parse(&layer).is_none() {
        return Err(usage(format!(
            "Invalid value for '--layer': '{layer}' is not one of 'short-term', 'mid-term', \
             'long-term'."
        )));
    }
    let role = p.get("role").unwrap_or("general").to_string();
    if !["dev", "ops", "sec", "ai", "general"].contains(&role.as_str()) {
        return Err(usage(format!(
            "Invalid value for '--role': '{role}' is not one of 'dev', 'ops', 'sec', 'ai', \
             'general'."
        )));
    }
    let intensity = p.float("intensity", 0.0)?;
    let valence = p.float("valence", 0.0)?;
    let own = owner()?;
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, true)?;
    let mut fields = Map::new();
    fields.insert("layer".into(), Value::String(layer));
    fields.insert("role".into(), Value::String(role));
    fields.insert(
        "tags".into(),
        Value::Array(csv(p.get("tags")).into_iter().map(Value::String).collect()),
    );
    fields.insert(
        "emotional".into(),
        json!({
            "intensity": intensity,
            "valence": valence,
            "labels": csv(p.get("emotions")),
            "resonance_note": p.get("resonance").unwrap_or(""),
        }),
    );
    fields.insert(
        "source".into(),
        Value::String(p.get("source").unwrap_or("cli").to_string()),
    );
    if p.has("decompose") {
        fields.insert("decompose".into(), Value::Bool(true));
    }
    let content = p.pos.pop().unwrap_or_default();
    let title = p.pos.pop().unwrap_or_default();
    let scope = eng.scope(None);
    let snap = eng
        .snapshot(&title, &content, fields, &own, &scope)
        .map_err(|e| match e {
            SnapshotError::Unsupported(m) => bare(m),
            SnapshotError::Invalid(e) => click_err(e.to_string()),
            SnapshotError::Engine(e) => engine_fail(e),
        })?;
    let m = snap.memory;
    let _ = writeln!(out, "Snapshot saved: {}", m.id);
    let _ = writeln!(out, "  Layer: {}", m.layer.as_str());
    let _ = writeln!(out, "  Emotional: {}", m.emotional.signature());
    if snap.created && env_nonempty("SKMEMORY_PG_DSN").is_some() {
        let _ = writeln!(
            err,
            "note: pgvector and graph rows are not written by this command; run \
             `skmemory sync --vector --graph`"
        );
    }
    Ok(0)
}

fn reindex_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let spec = Spec {
        value: &["base"],
        flag: &["force", "no-vector"],
        positional: &[],
    };
    let p = parse(&spec, args)?;
    let own = owner()?;
    let dir = memory_dir(p.get("base"))?;
    let eng = open_engine(&dir, true)?;
    let r = eng
        .index()
        .reindex(eng.store(), p.has("force"), &own)
        .map_err(|e| click_err(e.to_string()))?;
    let _ = writeln!(out, "Indexed {} memories into SQLite.", r.indexed);
    Ok(0)
}

fn intentions_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(usage("Missing command."));
    };
    match sub.as_str() {
        "list" => {
            let spec = Spec {
                value: &["state", "base"],
                flag: &[],
                positional: &[],
            };
            let p = parse(&spec, rest)?;
            let own = owner()?;
            let dir = memory_dir(p.get("base"))?;
            let st = IntentionStore::open(&dir, &own).map_err(|e| click_err(e.to_string()))?;
            let rows: Vec<Value> = st
                .list(p.get("state"))
                .map_err(|e| click_err(e.to_string()))?
                .iter()
                .map(|i| serde_json::from_str(&i.dump_compact()).unwrap_or(Value::Null))
                .collect();
            let _ = writeln!(out, "{}", dumps_indent2(&Value::Array(rows)));
            Ok(0)
        }
        "done" => {
            let spec = Spec {
                value: &["base"],
                flag: &[],
                positional: &["intention_id"],
            };
            let p = parse(&spec, rest)?;
            let own = owner()?;
            let dir = memory_dir(p.get("base"))?;
            let st = IntentionStore::open(&dir, &own).map_err(|e| click_err(e.to_string()))?;
            match st
                .apply(&p.pos[0], "done", None, None)
                .map_err(|e| click_err(e.to_string()))?
            {
                None => Err(click_err(format!("intention not found: {}", p.pos[0]))),
                Some(i) => {
                    let _ = writeln!(out, "{}: {}", p.pos[0], i.state.as_str());
                    Ok(0)
                }
            }
        }
        other => Err(usage(format!("No such command '{other}'."))),
    }
}

fn ledger_cmd(args: &[String], out: &mut dyn Write) -> Res<i32> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(usage("Missing command."));
    };
    if sub != "top" {
        return Err(usage(format!("No such command '{sub}'.")));
    }
    let spec = Spec {
        value: &["limit", "as-of", "base"],
        flag: &[],
        positional: &[],
    };
    let p = parse(&spec, rest)?;
    let limit = p.int("limit", 20)?.max(0) as usize;
    let as_of = match p.get("as-of") {
        Some(s) => Some(Dt::parse(s).ok_or_else(|| bare(format!("Invalid isoformat string: '{s}'")))?),
        None => None,
    };
    let own = owner()?;
    let dir = memory_dir(p.get("base"))?;
    let rows = Ledger::new(&dir, None, &own).top(as_of, limit);
    if rows.is_empty() {
        let _ = writeln!(out, "No ledger events.");
        return Ok(0);
    }
    let _ = writeln!(
        out,
        "{:>2}  {:<34} {:>7} {:>6} {:>4} {:>8}  flags",
        "#", "skill@version", "net", "follow", "srv", "imp f/s"
    );
    for s in rows {
        let mut flags: Vec<&str> = Vec::new();
        if s.anti_pattern {
            flags.push("anti-pattern");
        }
        if let Some(f) = s.flag {
            flags.push(f);
        }
        let imp = format!("{}/{}", s.imported_followed, s.imported_served);
        let _ = writeln!(
            out,
            "{:>2}  {:<34} {:>+7.2} {:>6.2} {:>4} {:>8}  {}",
            s.rank,
            s.key(),
            s.net,
            s.follow_rate,
            s.served,
            imp,
            flags.join(",")
        );
    }
    Ok(0)
}

const HELP: &str = "Usage: skmem [OPTIONS] COMMAND [ARGS]...\n\n  SKMemory engine CLI (Rust).\n\nOptions:\n  --base DIR     Memory directory (default: the active agent's memory)\n  --no-vector    Flat JSON + SQLite only (no Postgres arms)\n  --version      Show the version and exit.\n  --help         Show this message and exit.\n\nCommands:\n  health      Check memory system health.\n  intentions  list | done\n  ledger      top\n  list        List stored memories.\n  recall      Retrieve a specific memory by ID (alias: show).\n  reindex     Rebuild the SQLite index from JSON files on disk.\n  search      Search memories by text or meaning.\n  snapshot    Take a polaroid: capture a moment as a memory.\n";

/// Run the CLI. Returns the process exit status.
pub fn run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    // Global options may precede the command.
    let mut globals: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() && args[i].starts_with("--") {
        let a = &args[i];
        let name = a.trim_start_matches("--");
        let (name, inline) = match name.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (name, None),
        };
        match name {
            "version" => {
                let _ = writeln!(out, "skmem, version {}", crate::version());
                return 0;
            }
            "help" => {
                let _ = write!(out, "{HELP}");
                return 0;
            }
            n if GLOBAL_FLAG.contains(&n) => {
                if n != "ai" {
                    globals.push(a.clone());
                }
                i += 1;
            }
            n if GLOBAL_VALUE.contains(&n) => {
                globals.push(a.clone());
                i += 1;
                if inline.is_none() {
                    if let Some(v) = args.get(i) {
                        globals.push(v.clone());
                        i += 1;
                    }
                }
            }
            _ => break,
        }
    }
    let Some((cmd, rest)) = args[i..].split_first() else {
        let _ = write!(err, "{HELP}");
        return 2;
    };
    let mut full: Vec<String> = globals;
    full.extend(rest.iter().cloned());
    let res = match cmd.as_str() {
        "search" => search_cmd(&full, out, err),
        "recall" | "show" => recall_cmd(&full, out),
        "list" => list_cmd(&full, out),
        "health" => health_cmd(&full, out),
        "snapshot" => snapshot_cmd(&full, out, err),
        "reindex" => reindex_cmd(&full, out),
        "intentions" => {
            let (g, r) = split_group(&full);
            intentions_cmd(&[r, g].concat(), out)
        }
        "ledger" => {
            let (g, r) = split_group(&full);
            ledger_cmd(&[r, g].concat(), out)
        }
        other => Err(usage(format!("No such command '{other}'."))),
    };
    match res {
        Ok(code) => code,
        Err(f) => {
            if f.bare {
                let _ = writeln!(err, "{}", f.msg);
            } else {
                if f.code == 2 {
                    let _ = writeln!(err, "Usage: skmem {cmd} [OPTIONS]");
                    let _ = writeln!(err, "Try 'skmem {cmd} --help' for help.\n");
                }
                let _ = writeln!(err, "Error: {}", f.msg);
            }
            f.code
        }
    }
}

/// For group commands the subcommand name must stay first: pull the leading
/// non-option word out of the globals+rest list and put the options after it.
fn split_group(all: &[String]) -> (Vec<String>, Vec<String>) {
    // `all` is `globals... rest...` where rest starts with the subcommand; the
    // globals are `--base X` / `--no-vector` and are re-appended after it.
    let mut g = Vec::new();
    let mut i = 0;
    while i < all.len() {
        match all[i].as_str() {
            "--base" => {
                g.push(all[i].clone());
                if let Some(v) = all.get(i + 1) {
                    g.push(v.clone());
                }
                i += 2;
            }
            s if s.starts_with("--base=") => {
                g.push(all[i].clone());
                i += 1;
            }
            "--no-vector" => i += 1,
            _ => break,
        }
    }
    (g, all[i..].to_vec())
}
