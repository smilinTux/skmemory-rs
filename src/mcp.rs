//! `skmem-mcp`: the MCP server (stdio) over the Rust [`Engine`] (SKMEM-R-09).
//!
//! It speaks MCP's JSON-RPC 2.0 over newline-delimited stdio, hand-rolled on
//! `serde_json` (no MCP crate, no async runtime). The tool names, descriptions and
//! input schemas are the Python server's (`skmemory/mcp_server.py`), generated into
//! [`schemas`] by `scripts/gen_mcp_tool_fixture.py` and checked against the
//! vendored fixture by `tests/mcp_schema_parity.rs`. Results are the same JSON
//! text the Python server returns (`json.dumps(indent=2)` for a result, a compact
//! `{"error": ...}` for an error), in one `text` content item.
//!
//! # Tools served and tools left to Python
//!
//! Served: [`SERVED`], the tools the engine backs end to end (store, search,
//! recall, history, list, remind, intentions, intention_done). Every other
//! Python tool is NOT listed by `tools/list` and answers a call with a clear
//! "served by the Python skmemory MCP server" error, except that an owner-only
//! tool called by a declared reader is refused with Python's own sentence first,
//! so a reader never learns a difference. See `docs/MCP.md`.
//!
//! # Reader policy (identical to Python)
//!
//! `SKMEMORY_READER` is a ceiling; a per-call `reader` argument can only narrow
//! it ([`Engine::scope`]). Owner-only tools are refused to any declared reader
//! (per-call or env) with `"<tool> is owner-only and cannot be used by a declared
//! reader"`. A hidden memory id reads as `Memory not found: <id>`, exactly like a
//! missing one, and links to hidden memories are dropped from returned rows.
//!
//! Known gap: Python lets a process under a `SKMEMORY_READER` ceiling call
//! `memory_store` (tier `self-generated`). The engine's writes need an
//! [`OwnerScope`], which refuses under a ceiling, so the Rust server refuses that
//! one call and says so.

mod schemas;

use crate::audience::{OwnerScope, ReaderScope};
use crate::engine::{Arms, Engine, EngineError, EngineOptions, SearchRequest};
use crate::env::Env;
use crate::intentions::{id_ok, parse_due_at, Intention, IntentionStore, Kind};
use crate::model::{py_strip, Layer, Memory};
use crate::pyjson::{dumps_compact, dumps_indent2};
use crate::pytime::Dt;
use crate::snapshot::SnapshotError;
use crate::validity::normalize_as_of;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// The tools this server lists and runs, in the Python server's order.
pub const SERVED: [&str; 8] = [
    "memory_store",
    "memory_search",
    "memory_recall",
    "memory_history",
    "memory_list",
    "memory_remind",
    "memory_intentions",
    "memory_intention_done",
];

/// Python tools this server does not run (the engine does not back them with
/// Python's result shape). The bool is "owner-only" (`TOOL_READER_POLICY`).
pub const NOT_SERVED: [(&str, bool); 18] = [
    ("memory_expand", false),
    ("memory_check_duplicate", false),
    ("memory_forget", true),
    ("memory_promote", true),
    ("memory_consolidate", true),
    ("memory_context", false),
    ("memory_export", true),
    ("memory_import", true),
    ("memory_health", false),
    ("memory_graph", false),
    ("memory_synthesize_daily", true),
    ("memory_synthesize_dreams", true),
    ("memory_auto_context", false),
    ("memory_verify", true),
    ("memory_audit", true),
    ("memory_extract", false),
    ("memory_save_session", false),
    ("memory_pre_compact", false),
];

/// Owner-only among the served tools.
const OWNER_ONLY_SERVED: [&str; 3] = [
    "memory_remind",
    "memory_intentions",
    "memory_intention_done",
];

/// Tools that gain a `reader` property (Python `_READER_TOOLS`).
const READER_TOOLS: [&str; 8] = [
    "memory_check_duplicate",
    "memory_recall",
    "memory_expand",
    "memory_history",
    "memory_list",
    "memory_context",
    "memory_health",
    "memory_graph",
];

/// Python's `MAX_CHAIN`.
const MAX_CHAIN: usize = 32;
/// Python's `SEARCH_HIT_DEBOUNCE_S`.
const SEARCH_HIT_DEBOUNCE_S: f64 = 3600.0;
const SNIPPET_CHARS: usize = 160;
/// MCP revisions this server can speak (newest first).
const PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// The tool list: name, description, inputSchema, as the Python server lists them.
pub fn tool_list() -> Vec<Value> {
    schemas::SCHEMAS
        .iter()
        .map(|(_, text)| serde_json::from_str(text).expect("generated schema is valid JSON"))
        .collect()
}

/// The names of every Python tool, served or not.
pub fn all_tool_names() -> Vec<&'static str> {
    SERVED
        .iter()
        .copied()
        .chain(NOT_SERVED.iter().map(|(n, _)| *n))
        .collect()
}

/// Whether `tool` is owner-only (`TOOL_READER_POLICY`), served or not.
pub fn is_owner_only(tool: &str) -> bool {
    OWNER_ONLY_SERVED.contains(&tool) || NOT_SERVED.iter().any(|(n, o)| *n == tool && *o)
}

/// Tools whose schema carries the shared `reader` property in Python.
pub fn reader_tools() -> &'static [&'static str] {
    &READER_TOOLS
}

// ---------------------------------------------------------------------------
// where the store is
// ---------------------------------------------------------------------------

/// The agent selectors, in Python's order.
const SELECTORS: [&str; 3] = ["SKAGENT", "SKCAPSTONE_AGENT", "SKMEMORY_AGENT"];

/// The agent to open, from the environment variables in `get`, with no
/// guessing: every selector that is set must agree, else `SK_DEFAULT_AGENT`.
/// (Python also accepts a sole installed agent and checks the profile registry;
/// this server asks for an explicit declaration instead.)
pub fn resolve_agent(get: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    let set: BTreeSet<String> = SELECTORS
        .iter()
        .filter_map(|k| get(k))
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    match set.len() {
        1 => Ok(set.into_iter().next().unwrap_or_default()),
        0 => get("SK_DEFAULT_AGENT")
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                "no active agent: set SKAGENT (or SKCAPSTONE_AGENT / SKMEMORY_AGENT / \
                 SK_DEFAULT_AGENT), or pass --memory-dir"
                    .to_string()
            }),
        _ => Err(
            "conflicting agent selectors (SKAGENT, SKCAPSTONE_AGENT, SKMEMORY_AGENT): \
                  set them all to the same agent or none"
                .to_string(),
        ),
    }
}

/// The memory directory for the agent, from `get`: `SKMEMORY_HOME/memory` when set
/// (the agent home, as Python's `config.py` reads it), else
/// `$SKCAPSTONE_HOME/agents/<agent>/memory`, else `~/.skcapstone/agents/<agent>/memory`.
pub fn resolve_memory_dir(get: &dyn Fn(&str) -> Option<String>) -> Result<PathBuf, String> {
    let nonempty = |k: &str| get(k).filter(|v| !v.is_empty());
    if let Some(h) = nonempty("SKMEMORY_HOME") {
        return Ok(PathBuf::from(h).join("memory"));
    }
    let agent = resolve_agent(get)?;
    let base = match nonempty("SKCAPSTONE_HOME") {
        Some(h) => PathBuf::from(h).join("agents"),
        None => PathBuf::from(nonempty("HOME").ok_or("HOME is not set")?)
            .join(".skcapstone")
            .join("agents"),
    };
    Ok(base.join(agent).join("memory"))
}

/// The agent a store belongs to: `<...>/agents/<agent>/memory`.
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
fn build_arms(agent: Option<&str>, read_only: bool) -> Arms {
    use crate::engine::pg_arms::{attach, PgConfig};
    use crate::engine::Attach;
    let Some(cfg) = PgConfig::from_env(read_only) else {
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
fn build_arms(_agent: Option<&str>, _read_only: bool) -> Arms {
    Arms::none()
}

// ---------------------------------------------------------------------------
// the server
// ---------------------------------------------------------------------------

/// An MCP server over one [`Engine`].
pub struct Server {
    engine: Engine,
}

type ToolResult = Result<Value, String>;

fn text_content(text: String) -> Value {
    json!({"content": [{"type": "text", "text": text}]})
}

fn ok_json(v: &Value) -> Value {
    text_content(dumps_indent2(v))
}

fn err_json(msg: &str) -> Value {
    text_content(dumps_compact(&json!({"error": msg})))
}

/// `str(KeyError(key))`, which is what the Python server reports for a missing
/// required argument.
fn missing(key: &str) -> String {
    format!("'{key}'")
}

fn req_str<'a>(args: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    match args.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(Value::Null) | None => Err(missing(key)),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

/// `arguments.get(key)` as an optional non-empty string (Python `or None`).
fn opt_str<'a>(args: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// `int(arguments.get(key, default))`.
fn int_arg(args: &Map<String, Value>, key: &str, default: i64) -> Result<i64, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| format!("{key} must be an integer")),
        Some(Value::Bool(b)) => Ok(i64::from(*b)),
        Some(Value::String(s)) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("invalid literal for int() with base 10: '{s}'")),
        Some(_) => Err(format!("{key} must be an integer")),
    }
}

fn str_list(args: &Map<String, Value>, key: &str) -> Vec<String> {
    match args.get(key) {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `disclosure.snippet`.
fn snippet(m: &Memory) -> String {
    let body = flat(if m.summary.is_empty() {
        &m.content
    } else {
        &m.summary
    });
    if body.chars().count() <= SNIPPET_CHARS {
        return body;
    }
    let cut: String = body.chars().take(SNIPPET_CHARS - 3).collect();
    format!("{}...", cut.trim_end())
}

/// `disclosure.parse_source_ref(ref) is not None`.
fn expandable(source_ref: &str) -> bool {
    let r = py_strip(source_ref);
    let Some(rest) = r.strip_prefix("session:") else {
        return false;
    };
    let (sid, tail) = match rest.split_once('@') {
        Some((s, t)) => (s, Some(t)),
        None => (rest, None),
    };
    let b = sid.as_bytes();
    let sid_ok = !b.is_empty()
        && b.len() <= 201
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'));
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    let tail_ok = match tail {
        None => true,
        Some(t) => match t.strip_prefix('L') {
            Some(n) => digits(n),
            None => t
                .split_once('-')
                .is_some_and(|(a, b)| digits(a) && digits(b)),
        },
    };
    sid_ok && tail_ok
}

/// `validity.superseded_by_map`, restricted to successors the reader may see.
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

/// `tiers.tier_for_explicit_store`.
pub fn tier_for_explicit_store(source: &str, owner: bool) -> &'static str {
    const GENERATED: [&str; 8] = [
        "skwhisper",
        "journal-synthesis",
        "dreaming-engine",
        "consolidation",
        "claude-code-hook",
        "mcp:memory_save_session",
        "mcp:memory_pre_compact",
        "observation",
    ];
    const TOOL: [&str; 5] = ["tool", "tool-output", "command", "shell", "task-pack"];
    let rank = |t: &str| match t {
        "user-stated" => 3,
        "tool-output" => 2,
        "web" => 1,
        _ => 0,
    };
    let base = if owner {
        "user-stated"
    } else {
        "self-generated"
    };
    let label = crate::env::casefold(py_strip(source));
    let cand = if GENERATED.contains(&label.as_str()) {
        "self-generated"
    } else if TOOL.contains(&label.as_str()) {
        "tool-output"
    } else if ["web", "http", "fetch", "browser", "search"]
        .iter()
        .any(|p| label.starts_with(p))
    {
        "web"
    } else {
        return base;
    };
    if rank(cand) < rank(base) {
        cand
    } else {
        base
    }
}

fn engine_msg(e: &EngineError) -> String {
    e.to_string()
}

impl Server {
    /// Open the store at `memory_dir`, with the Postgres arms when `pg` is on and
    /// `SKMEMORY_PG_DSN` is set. A process without owner rights opens read-only.
    pub fn open(memory_dir: &Path) -> Result<Server, String> {
        let read_only = OwnerScope::from_process().is_err();
        let agent = agent_of(memory_dir);
        let arms = build_arms(agent.as_deref(), read_only);
        let cfg = std::env::var_os("SKMEMORY_AUDIENCE_CONFIG").map(PathBuf::from);
        let engine = Engine::open(
            memory_dir,
            EngineOptions {
                audience_config: cfg,
                fts: true,
                read_only: Some(read_only),
                arms,
            },
        )
        .map_err(|e| engine_msg(&e))?;
        Ok(Server { engine })
    }

    /// A server over an already-open engine (tests).
    pub fn with_engine(engine: Engine) -> Server {
        Server { engine }
    }

    fn visible(&self, id: &str, scope: &ReaderScope) -> bool {
        self.engine.store().load(id, scope).ok().flatten().is_some()
    }

    fn may_reinforce(&self, scope: &ReaderScope) -> bool {
        scope.unrestricted() && !self.engine.is_read_only() && OwnerScope::from_process().is_ok()
    }

    fn owner_rw(&self) -> Result<OwnerScope, String> {
        let own = OwnerScope::from_process().map_err(|_| {
            "a declared reader cannot write: the engine's writes are owner only".to_string()
        })?;
        if self.engine.is_read_only() {
            return Err("this engine was opened read-only".into());
        }
        Ok(own)
    }

    /// `ReaderScope` for a call's `reader` argument.
    fn scope_of(&self, args: &Map<String, Value>) -> ReaderScope {
        self.engine
            .scope(args.get("reader").and_then(Value::as_str))
    }

    /// `memory_to_dict` with links to memories the reader cannot see removed
    /// (Python `_reader_memory_dict` / `redact_links`).
    fn reader_row(&self, m: &Memory, scope: &ReaderScope) -> Value {
        if scope.unrestricted() {
            return m.to_value();
        }
        let mut m = m.clone();
        let mut cache: HashMap<String, bool> = HashMap::new();
        let mut vis = |id: &str| -> bool {
            if id.is_empty() {
                return false;
            }
            *cache
                .entry(id.to_string())
                .or_insert_with(|| self.visible(id, scope))
        };
        scope.redact_links(&mut m, &mut vis);
        if m.parent_id.is_none() {
            m.metadata.remove("parent_relation");
        }
        let hide_promoted = m
            .metadata
            .get("promoted_id")
            .is_some_and(|v| !v.as_str().is_some_and(&mut vis));
        if hide_promoted {
            m.metadata.remove("promoted_id");
        }
        if let Some(Value::Array(a)) = m.metadata.get("possible_conflict_with").cloned() {
            let kept: Vec<Value> = a
                .into_iter()
                .filter(|v| v.as_str().is_some_and(&mut vis))
                .collect();
            if kept.is_empty() {
                m.metadata.remove("possible_conflict_with");
            } else {
                m.metadata
                    .insert("possible_conflict_with".into(), Value::Array(kept));
            }
        }
        if let Some(Value::Array(a)) = m.metadata.get("chunk_memory_ids").cloned() {
            let kept: Vec<Value> = a
                .into_iter()
                .filter(|v| v.as_str().is_some_and(&mut vis))
                .collect();
            m.metadata
                .insert("chunk_memory_ids".into(), Value::Array(kept));
        }
        let hide_triage = m
            .metadata
            .get("write_triage")
            .and_then(Value::as_object)
            .is_some_and(|t| {
                !t.get("target")
                    .and_then(Value::as_str)
                    .is_some_and(&mut vis)
            });
        if hide_triage {
            m.metadata.remove("write_triage");
        }
        if let Some(Value::Object(d)) = m.metadata.get("decomposition").cloned() {
            if d.get("parent_id").is_some_and(|p| !p.is_null() && p != "") {
                let ok = d
                    .get("parent_id")
                    .and_then(Value::as_str)
                    .is_some_and(&mut vis);
                if !ok {
                    let kept: Map<String, Value> =
                        d.into_iter().filter(|(k, _)| k != "parent_id").collect();
                    m.metadata
                        .insert("decomposition".into(), Value::Object(kept));
                }
            }
        }
        // An observation names its evidence. Hidden ids are dropped; the text is
        // cut back to its statement (fail closed: Python keeps the visible quotes,
        // this server does not rebuild them).
        if let Some(Value::Object(obs)) = m.metadata.get("observation").cloned() {
            if let Some(Value::Array(ev)) = obs.get("evidence_ids") {
                let kept: Vec<Value> = ev
                    .iter()
                    .filter(|v| v.as_str().is_some_and(&mut vis))
                    .cloned()
                    .collect();
                let lost = kept.len() != ev.len();
                let hidden: Vec<String> = ev
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|i| !kept.iter().any(|k| k.as_str() == Some(i)))
                    .map(str::to_string)
                    .collect();
                let mut o = obs.clone();
                if o.contains_key("proof_count") {
                    o.insert("proof_count".into(), Value::from(kept.len()));
                }
                o.insert("evidence_ids".into(), Value::Array(kept));
                m.metadata.insert("observation".into(), Value::Object(o));
                if lost {
                    let head = m.content.split("\n\n").next().unwrap_or("").to_string();
                    let leaks = |t: &str| hidden.iter().any(|h| t.contains(h.as_str()));
                    m.content = if leaks(&head) {
                        "Observation".into()
                    } else {
                        head
                    };
                    if leaks(&m.title) {
                        m.title = "Observation".into();
                    }
                }
            }
        }
        m.to_value()
    }

    /// Owner-only gate: `Some(refusal)` for a declared reader.
    fn owner_only(&self, args: &Map<String, Value>, tool: &str) -> Option<Value> {
        if self.scope_of(args).unrestricted() {
            return None;
        }
        Some(err_json(&format!(
            "{tool} is owner-only and cannot be used by a declared reader"
        )))
    }

    /// Run one tool call. Never panics on bad input; every failure is an
    /// `{"error": ...}` content item, as in the Python server.
    pub fn call_tool(&self, name: &str, arguments: &Value) -> Value {
        let empty = Map::new();
        let args = arguments.as_object().unwrap_or(&empty);
        if is_owner_only(name) {
            if let Some(refused) = self.owner_only(args, name) {
                return refused;
            }
        }
        let res: ToolResult = match name {
            "memory_store" => self.t_store(args),
            "memory_search" => self.t_search(args),
            "memory_recall" => self.t_recall(args),
            "memory_history" => self.t_history(args),
            "memory_list" => self.t_list(args),
            "memory_remind" => self.t_remind(args),
            "memory_intentions" => self.t_intentions(args),
            "memory_intention_done" => self.t_intention_done(args),
            n if NOT_SERVED.iter().any(|(t, _)| *t == n) => Err(format!(
                "{n} is not served by skmem-mcp; use the Python skmemory MCP server for it"
            )),
            n => Err(format!("Unknown tool: {n}")),
        };
        match res {
            Ok(v) => v,
            Err(e) => err_json(&e),
        }
    }

    // ----- memory_store -----------------------------------------------------

    fn t_store(&self, args: &Map<String, Value>) -> ToolResult {
        let title = req_str(args, "title")?;
        let content = req_str(args, "content")?;
        let layer = opt_layer(args.get("layer").and_then(Value::as_str)).unwrap_or(Layer::Short);
        let source = args.get("source").and_then(Value::as_str).unwrap_or("mcp");
        let scope = self.scope_of(args);
        let own = self.owner_rw()?;
        let mut fields = Map::new();
        fields.insert("layer".into(), Value::from(layer.as_str()));
        fields.insert("tags".into(), json!(str_list(args, "tags")));
        fields.insert("source".into(), Value::from(source));
        fields.insert(
            "source_tier".into(),
            Value::from(tier_for_explicit_store(source, scope.unrestricted())),
        );
        let snap = self
            .engine
            .snapshot(title, content, fields, &own, &scope)
            .map_err(|e| match e {
                SnapshotError::Unsupported(m) => m,
                SnapshotError::Invalid(e) => e.to_string(),
                SnapshotError::Engine(e) => engine_msg(&e),
            })?;
        Ok(ok_json(
            &json!({"memory_id": snap.memory.id, "stored": true}),
        ))
    }

    // ----- memory_search ----------------------------------------------------

    fn t_search(&self, args: &Map<String, Value>) -> ToolResult {
        let query = req_str(args, "query")?;
        let limit = int_arg(args, "limit", 10)?.max(0) as usize;
        let tags = str_list(args, "tags");
        let layer = opt_str(args, "layer");
        if let Some(l) = layer {
            if Layer::parse(l).is_none() {
                return Err(format!("unknown layer {l:?}"));
            }
        }
        let source = opt_str(args, "source");
        let as_of = match opt_str(args, "as_of") {
            Some(a) => normalize_as_of(a).map_err(|e| e.to_string())?,
            None => None,
        };
        let detail = args.get("detail").and_then(Value::as_str);
        let scope = self.scope_of(args);
        let out = self
            .engine
            .search(
                &SearchRequest {
                    query,
                    limit,
                    layer,
                    tags: &tags,
                    source,
                    context_tags: None,
                    as_of: as_of.as_deref(),
                    now: None,
                },
                &scope,
                &Env::process(),
            )
            .map_err(|e| engine_msg(&e))?;
        let mems: Vec<Memory> = out.memories.into_iter().take(limit).collect();
        if self.may_reinforce(&scope) {
            let ids: Vec<&str> = mems.iter().map(|m| m.id.as_str()).collect();
            self.engine
                .index()
                .bump_access(&ids, None, Some(SEARCH_HIT_DEBOUNCE_S), &scope);
        }
        let sup = superseded_map(&mems, &mut |id| self.visible(id, &scope));
        let mut rows: Vec<Value> = mems
            .iter()
            .map(|m| {
                let mut v = self.reader_row(m, &scope);
                if let Some(o) = v.as_object_mut() {
                    o.insert("superseded".into(), Value::Bool(sup.contains_key(&m.id)));
                    o.insert(
                        "superseded_by".into(),
                        sup.get(&m.id)
                            .map_or(Value::Null, |s| Value::from(s.as_str())),
                    );
                }
                v
            })
            .collect();
        if detail == Some("index") {
            let est = |v: &Value| dumps_compact(v).len().div_ceil(4);
            rows = mems
                .iter()
                .zip(rows.iter())
                .enumerate()
                .map(|(i, (m, full))| {
                    let rank = (i + 1) as f64;
                    let mut row = json!({
                        "id": m.id,
                        "title": m.title,
                        "date": m.created_at,
                        "layer": m.layer.as_str(),
                        "score": (10000.0 / rank).round() / 10000.0,
                        "score_kind": "rank",
                        "est_basis": "chars/4",
                        "snippet": snippet(m),
                        "source_ref": m.source_ref,
                        "expandable": expandable(&m.source_ref),
                        "superseded": full["superseded"],
                        "superseded_by": full["superseded_by"],
                    });
                    let t = est(&row);
                    if let Some(o) = row.as_object_mut() {
                        o.insert("est_tokens".into(), t.into());
                        o.insert("est_tokens_full".into(), est(full).into());
                    }
                    row
                })
                .collect();
        }
        if let Some(d) = out.report.degraded {
            return Ok(ok_json(&json!({"results": rows, "degraded": d})));
        }
        Ok(ok_json(&Value::Array(rows)))
    }

    // ----- memory_recall ----------------------------------------------------

    fn t_recall(&self, args: &Map<String, Value>) -> ToolResult {
        let id = req_str(args, "memory_id")?;
        let scope = self.scope_of(args);
        let loaded = self
            .engine
            .store()
            .load(id, &scope)
            .map_err(|e| e.to_string())?;
        let Some(mut m) = loaded else {
            return Err(format!("Memory not found: {id}"));
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
        if self.may_reinforce(&scope) {
            self.engine.index().bump_access(&[id], None, None, &scope);
        }
        Ok(ok_json(&self.reader_row(&m, &scope)))
    }

    // ----- memory_list ------------------------------------------------------

    fn t_list(&self, args: &Map<String, Value>) -> ToolResult {
        // An unknown layer string is "no layer filter", as Python's `_LAYER_MAP.get`.
        let layer = opt_layer(args.get("layer").and_then(Value::as_str));
        let tags = str_list(args, "tags");
        let limit = int_arg(args, "limit", 50)?.max(0) as usize;
        let scope = self.scope_of(args);
        let listing = self.engine.store().list(layer, &scope);
        let rows: Vec<Value> = listing
            .memories
            .iter()
            .filter(|m| tags.iter().all(|t| m.tags.contains(t)))
            .take(limit)
            .map(|m| self.reader_row(m, &scope))
            .collect();
        Ok(ok_json(&Value::Array(rows)))
    }

    // ----- memory_history ---------------------------------------------------

    fn raw_load(&self, id: &str) -> Option<Memory> {
        self.engine
            .store()
            .find_file(id)
            .and_then(|p| self.engine.store().read_path(&p).ok())
    }

    fn successors(&self, cur: &Memory) -> Result<Vec<String>, String> {
        let mut ids: Vec<String> = Vec::new();
        if let Some(s) = cur
            .metadata
            .get("superseded_by")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_str)
        {
            ids.push(s.to_string());
        }
        ids.extend(
            self.engine
                .index()
                .ids_superseding(&cur.id)
                .map_err(|e| e.to_string())?,
        );
        let mut seen = HashSet::new();
        Ok(ids
            .into_iter()
            .filter(|i| !i.is_empty() && seen.insert(i.clone()))
            .collect())
    }

    fn t_history(&self, args: &Map<String, Value>) -> ToolResult {
        let memory_id = req_str(args, "memory_id")?;
        let scope = self.scope_of(args);
        let seen_ok = |m: &Memory| scope.allows_memory(m);
        let Some(start) = self.raw_load(memory_id).filter(seen_ok) else {
            return Err(format!("Memory not found: {memory_id}"));
        };
        let mut chain: Vec<Memory> = vec![start.clone()];
        let mut seen: HashSet<String> = HashSet::from([start.id.clone()]);
        let mut cur = start.clone();
        for _ in 0..MAX_CHAIN {
            let prev = cur
                .supersedes_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .and_then(|s| self.raw_load(s));
            match prev {
                Some(p) if !seen.contains(&p.id) && seen_ok(&p) => {
                    seen.insert(p.id.clone());
                    chain.insert(0, p.clone());
                    cur = p;
                }
                _ => break,
            }
        }
        let mut cur = start;
        let mut hidden_next = false;
        for _ in 0..MAX_CHAIN {
            let nxt = self
                .successors(&cur)?
                .iter()
                .filter_map(|i| self.raw_load(i))
                .find(|m| !seen.contains(&m.id));
            let Some(nxt) = nxt else { break };
            if !seen_ok(&nxt) {
                hidden_next = true;
                break;
            }
            seen.insert(nxt.id.clone());
            chain.push(nxt.clone());
            cur = nxt;
        }
        let keep: BTreeSet<&str> = chain.iter().map(|m| m.id.as_str()).collect();
        let masked = hidden_next.then(|| chain[chain.len() - 1].id.clone());
        let opt = |s: &Option<String>| s.clone().map_or(Value::Null, Value::String);
        let rows: Vec<Value> = chain
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let is_masked = masked.as_deref() == Some(m.id.as_str());
                let named = m
                    .metadata
                    .get("superseded_by")
                    .and_then(|v| v.get("id"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty());
                let sup_by = match named {
                    Some(s) if keep.contains(s) => Some(s.to_string()),
                    _ => chain.get(i + 1).map(|n| n.id.clone()),
                };
                json!({
                    "id": m.id,
                    "title": m.title,
                    "layer": m.layer.as_str(),
                    "created_at": m.created_at,
                    "valid_from": opt(&m.valid_from),
                    "valid_to": if is_masked { Value::Null } else { opt(&m.valid_to) },
                    "supersedes_id": m.supersedes_id.as_deref()
                        .filter(|s| keep.contains(s))
                        .map_or(Value::Null, |s| Value::String(s.to_string())),
                    "superseded_by": opt(&sup_by),
                    "current": m.valid_to.is_none() || is_masked,
                    "requested": m.id == memory_id,
                })
            })
            .collect();
        Ok(ok_json(&Value::Array(rows)))
    }

    // ----- intentions (owner only; the gate already ran) ----------------------

    fn intention_store(&self) -> Result<IntentionStore, String> {
        let own = self.owner_rw()?;
        IntentionStore::open(self.engine.root(), &own).map_err(|e| e.to_string())
    }

    fn t_remind(&self, args: &Map<String, Value>) -> ToolResult {
        let title = flat(args.get("title").and_then(Value::as_str).unwrap_or(""));
        if title.is_empty() {
            return Err("title is required".into());
        }
        if title.chars().count() > 500 {
            return Err("title is too long (500 characters max)".into());
        }
        let due_text = args.get("due_at").and_then(Value::as_str).unwrap_or("");
        let due = parse_due_at(due_text, None).map_err(|e| e.to_string())?;
        let origin = opt_str(args, "origin_memory_id");
        if let Some(o) = origin {
            if !id_ok(o) {
                return Err(format!("invalid intention id: {}", py_repr(o)));
            }
        }
        let interval = match args.get("interval_s") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s.is_empty() => None,
            Some(_) => Some(int_arg(args, "interval_s", 0)?),
        };
        if interval.is_some_and(|s| s < 60) {
            return Err("interval_s must be at least 60 seconds".into());
        }
        let id: String = crate::util::uuid4()
            .replace('-', "")
            .chars()
            .take(16)
            .collect();
        let mut i = Intention::new(&id, Kind::Time, &title);
        i.due_at = Some(due);
        i.interval_s = interval;
        i.origin_memory_id = origin.map(str::to_string);
        i.channel = match args.get("channel") {
            Some(Value::String(c)) if !c.is_empty() => c.clone(),
            _ => "inline".to_string(),
        };
        let st = self.intention_store()?;
        st.save(&i).map_err(|e| e.to_string())?;
        Ok(ok_json(
            &json!({"id": i.id, "state": i.state.as_str(), "due_at": due.isoformat()}),
        ))
    }

    fn t_intentions(&self, args: &Map<String, Value>) -> ToolResult {
        let st = self.intention_store()?;
        let rows: Vec<Value> = st
            .list(opt_str(args, "state"))
            .map_err(|e| e.to_string())?
            .iter()
            .map(|i| serde_json::from_str(&i.dump_compact()).unwrap_or(Value::Null))
            .collect();
        Ok(ok_json(&Value::Array(rows)))
    }

    fn t_intention_done(&self, args: &Map<String, Value>) -> ToolResult {
        let iid = args.get("id").and_then(Value::as_str).unwrap_or("");
        let st = self.intention_store()?;
        match st
            .apply(iid, "done", None, None)
            .map_err(|e| e.to_string())?
        {
            None => Err(format!("Intention not found: {iid}")),
            Some(i) => Ok(ok_json(&json!({"id": iid, "state": i.state.as_str()}))),
        }
    }

    // ----- JSON-RPC ----------------------------------------------------------

    /// Handle one JSON-RPC message. `None` for a notification (no reply).
    pub fn handle(&self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str);
        let Some(method) = method else {
            // A response or junk: nothing to answer.
            return id.map(|id| rpc_err(id, -32600, "Invalid Request"));
        };
        let id = id?; // a notification has no id and gets no reply
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => {
                let want = params.get("protocolVersion").and_then(Value::as_str);
                let version = want
                    .filter(|w| PROTOCOLS.contains(w))
                    .unwrap_or(PROTOCOLS[0]);
                json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "skmemory", "version": crate::version()},
                })
            }
            "ping" => json!({}),
            "tools/list" => {
                let tools = tool_list();
                json!({"tools": tools})
            }
            "tools/call" => {
                let Some(name) = params.get("name").and_then(Value::as_str) else {
                    return Some(rpc_err(id, -32602, "tools/call needs a tool name"));
                };
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                self.call_tool(name, &args)
            }
            _ => return Some(rpc_err(id, -32601, "Method not found")),
        };
        Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    /// Serve newline-delimited JSON-RPC until `input` ends.
    pub fn serve(&self, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let reply = match serde_json::from_str::<Value>(&line) {
                Ok(msg) => self.handle(&msg),
                Err(_) => Some(rpc_err(Value::Null, -32700, "Parse error")),
            };
            if let Some(r) = reply {
                let text = serde_json::to_string(&r).unwrap_or_default();
                writeln!(output, "{text}")?;
                output.flush()?;
            }
        }
        Ok(())
    }
}

fn rpc_err(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn opt_layer(s: Option<&str>) -> Option<Layer> {
    s.and_then(Layer::parse)
}

/// Python `repr(str)` for the ASCII ids this reports.
fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// Entry point of the `skmem-mcp` binary: `skmem-mcp [--memory-dir DIR]`.
pub fn run_main() -> i32 {
    let mut dir: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--memory-dir" => match it.next() {
                Some(d) => dir = Some(PathBuf::from(d)),
                None => {
                    eprintln!("skmem-mcp: --memory-dir needs a value");
                    return 2;
                }
            },
            "--version" => {
                println!("skmem-mcp {}", crate::version());
                return 0;
            }
            "-h" | "--help" => {
                println!(
                    "skmem-mcp [--memory-dir DIR]\n\nMCP server (stdio) over the skmem engine. \
                     The store is DIR, else $SKMEMORY_HOME/memory, else the active agent's \
                     memory/ (SKAGENT et al.)."
                );
                return 0;
            }
            other => {
                eprintln!("skmem-mcp: unknown argument {other}");
                return 2;
            }
        }
    }
    let dir = match dir {
        Some(d) => d,
        None => match resolve_memory_dir(&|k| std::env::var(k).ok()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skmem-mcp: {e}");
                return 2;
            }
        },
    };
    let server = match Server::open(&dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skmem-mcp: {e}");
            return 1;
        }
    };
    let stdin = std::io::stdin();
    match server.serve(stdin.lock(), std::io::stdout().lock()) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("skmem-mcp: {e}");
            1
        }
    }
}
