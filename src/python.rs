//! PyO3 bindings: the scoped skmem API for the R-08 shim (SKMEM-R-07B).
//!
//! Built only with the `python` feature. The surface is one class, [`Engine`],
//! plus `version()`. Results are plain dicts and lists whose keys are the
//! Python `MemoryStore` field names (see `docs/PYTHON_BINDINGS.md`).
//!
//! # Scope rules (carried over from R-06C / R-06D / P0-20)
//!
//! - No binding accepts, returns or names a scope object. `OwnerScope` and
//!   `ReaderScope` are never exposed to Python, and the `testing` feature
//!   (which holds the env-injecting constructors) cannot be combined with
//!   `extension-module` (`compile_error!` in `audience.rs`).
//! - Every call derives its scope from the PROCESS: reads use
//!   [`ReaderScope::from_process_with_config`] (the per-call `reader` can only
//!   narrow the `SKMEMORY_READER` ceiling), and owner-only calls use
//!   [`OwnerScope::from_process`], which refuses under a ceiling.
//! - Owner-only calls raise `PermissionError` under a reader ceiling.
//! - A hidden id reads exactly like a missing one: `None` or `[]`, never an
//!   error that says the id exists.
//!
//! Python can still change its own process environment (`os.environ`), the same
//! trust boundary the Python implementation has. The bindings add no way to
//! skip the environment, which is what the tests pin.

use crate::audience::{AudienceConfig, OwnerScope, ReaderScope};
use crate::decay::Usage;
use crate::env::Env;
use crate::flat::{Store, StoreError};
use crate::fusion::{run_arm, ArmInput, DEFAULT_ARM_CAP};
use crate::index::{wal_status, Index, SearchFilters, WAL_DEGRADED_BYTES};
use crate::intentions::{parse_due_at, Intention, IntentionStore, Kind};
use crate::ledger::{Ledger, Record, Score};
use crate::model::{Layer, Memory};
use crate::pytime::Dt;
use crate::search::{search, Candidate, SearchInput};
use crate::validity::{normalize_as_of, valid_at};
use pyo3::exceptions::{PyOSError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

/// Python's `SEARCH_HIT_DEBOUNCE_S`.
const SEARCH_HIT_DEBOUNCE_S: f64 = 3600.0;
/// Python's `MAX_CHAIN`.
const MAX_CHAIN: usize = 32;
const SNIPPET_CHARS: usize = 160;

// ---------------------------------------------------------------------------
// errors and JSON bridging
// ---------------------------------------------------------------------------

fn store_err(e: StoreError) -> PyErr {
    match e {
        StoreError::Io(e) => PyOSError::new_err(e.to_string()),
        StoreError::BadId(_) => PyValueError::new_err(e.to_string()),
        StoreError::Invalid(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

fn index_err(e: crate::index::IndexError) -> PyErr {
    match e {
        crate::index::IndexError::Io(e) => PyOSError::new_err(e.to_string()),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The caller is the owner, or `PermissionError`. Derived from the process
/// environment on every call: there is no way to pass a scope in.
fn owner() -> PyResult<OwnerScope> {
    OwnerScope::from_process().map_err(|e| PyPermissionError::new_err(e.to_string()))
}

/// A JSON value as the equivalent Python object (via `json.loads`, so floats
/// and big integers keep their text).
fn to_py(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    let text = serde_json::to_string(v).map_err(value_err)?;
    Ok(py.import("json")?.call_method1("loads", (text,))?.unbind())
}

/// A Python object as JSON (via `json.dumps(default=str)`, as Python's
/// `FileBackend.save` does).
fn from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json_mod = py.import("json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("default", py.get_type::<pyo3::types::PyString>())?;
    let text: String = json_mod
        .call_method("dumps", (obj,), Some(&kwargs))?
        .extract()?;
    serde_json::from_str(&text).map_err(value_err)
}

fn parse_layer(s: &str) -> PyResult<Layer> {
    Layer::parse(s).ok_or_else(|| value_err(format!("unknown layer {s:?}")))
}

/// `str | list[str] | None` as a list.
fn str_or_list(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(o) = obj else { return Ok(None) };
    if o.is_none() {
        return Ok(None);
    }
    if let Ok(s) = o.extract::<String>() {
        return Ok(Some(vec![s]));
    }
    Ok(Some(o.extract::<Vec<String>>()?))
}

fn forgotten_at() -> String {
    let (y, mo, d, h, mi, s, _) = Dt::now().to_utc().fields();
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}+0000")
}

fn score_value(s: &Score) -> Value {
    json!({
        "skill": s.skill, "version": s.version, "key": s.key(),
        "served": s.served, "followed": s.followed, "follow_rate": s.follow_rate,
        "net": s.net, "net_effect_sign": s.net_effect_sign,
        "anti_pattern": s.anti_pattern, "flag": s.flag, "rank": s.rank,
        "imported_served": s.imported_served, "imported_followed": s.imported_followed,
    })
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
    let r = crate::model::py_strip(source_ref);
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

// ---------------------------------------------------------------------------
// the engine
// ---------------------------------------------------------------------------

/// A memory store rooted at an agent's `memory/` directory (flat JSON files
/// plus the SQLite `index.db`).
///
/// Reads take an optional `reader` that can only narrow the process
/// `SKMEMORY_READER` ceiling. Writes and maintenance are owner only.
#[pyclass(frozen, module = "skmem_core")]
pub struct Engine {
    root: PathBuf,
    store: Store,
    index: Mutex<Index>,
    cfg: AudienceConfig,
}

impl Engine {
    /// The reader scope for this call. The per-call `reader` is resolved under
    /// the engine's audience config; the `SKMEMORY_READER` ceiling under the
    /// process one. See [`ReaderScope::from_process_with_config`].
    fn scope(&self, reader: Option<&str>) -> ReaderScope {
        ReaderScope::from_process_with_config(reader, &self.cfg)
    }

    fn idx(&self) -> PyResult<std::sync::MutexGuard<'_, Index>> {
        self.index
            .lock()
            .map_err(|_| PyRuntimeError::new_err("index lock poisoned"))
    }

    fn visible(&self, id: &str, scope: &ReaderScope) -> bool {
        self.store.load(id, scope).ok().flatten().is_some()
    }

    /// The fused search, returning the page of hydrated memories in rank order.
    #[allow(clippy::too_many_arguments)]
    fn run_search(
        &self,
        query: &str,
        limit: usize,
        layer: Option<&str>,
        tags: &[String],
        source: Option<&str>,
        context_tags: Option<&[String]>,
        as_of: Option<&str>,
        scope: &ReaderScope,
    ) -> PyResult<Vec<Memory>> {
        let pool = limit.max(limit * 3);
        let arm_limit = (pool * 3).max(DEFAULT_ARM_CAP);
        let rows = self
            .idx()?
            .search_lexical(
                query,
                arm_limit,
                &SearchFilters {
                    layer,
                    tags,
                    source,
                    context_tags,
                    as_of,
                },
                scope,
            )
            .map_err(index_err)?;
        let mut found: HashMap<String, Candidate> = HashMap::new();
        let mut mems: HashMap<String, Memory> = HashMap::new();
        let mut ids: Vec<String> = Vec::new();
        for r in rows {
            // Re-check on the hydrated memory: the flat file is the truth.
            let Some(m) = self.store.load(&r.id, scope).map_err(store_err)? else {
                continue;
            };
            let ok = layer.is_none_or(|l| m.layer.as_str() == l)
                && tags.iter().all(|t| m.tags.contains(t))
                && source.is_none_or(|s| s.is_empty() || m.source == s)
                && as_of.is_none_or(|a| {
                    valid_at(
                        m.valid_from.as_deref(),
                        Some(&m.created_at),
                        m.valid_to.as_deref(),
                        a,
                    )
                });
            if !ok {
                continue;
            }
            found.insert(m.id.clone(), Candidate::from_memory(&m));
            ids.push(m.id.clone());
            mems.insert(m.id.clone(), m);
        }
        let usage: HashMap<String, Usage> = {
            let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            self.idx()?
                .get_usage(&refs, scope)
                .into_iter()
                .map(|(k, (n, last))| {
                    (
                        k,
                        Usage {
                            access_count: n,
                            last_accessed: last,
                        },
                    )
                })
                .collect()
        };
        let cap = DEFAULT_ARM_CAP;
        let arms = vec![
            run_arm("semantic", ArmInput::Unattached, cap),
            run_arm("lexical", ArmInput::Unattached, cap),
            run_arm("lexical_local", ArmInput::Ids(ids), cap),
            run_arm(
                "graph",
                ArmInput::Disabled(
                    "disabled by config (SKMEMORY_GRAPH_ARM is off, the default)".into(),
                ),
                cap,
            ),
            run_arm("temporal", ArmInput::Unattached, cap),
        ];
        let input = SearchInput {
            query,
            scope,
            limit,
            arms: &arms,
            found: &found,
            has_pg_bm25: false,
            df: None,
            usage: Some(&usage),
            decay_as_of: Dt::now(),
        };
        let page = match search(&input, &Env::process()) {
            Some(t) => t.page,
            None => Vec::new(),
        };
        Ok(page.into_iter().filter_map(|i| mems.remove(&i)).collect())
    }

    /// The forward neighbours of `cur` in a supersession chain: the id its
    /// metadata names plus every record that names it in `supersedes_id`.
    fn successors(&self, cur: &Memory) -> PyResult<Vec<String>> {
        let mut ids: Vec<String> = Vec::new();
        if let Some(s) = cur
            .metadata
            .get("superseded_by")
            .and_then(|v| v.get("id"))
            .and_then(Value::as_str)
        {
            ids.push(s.to_string());
        }
        ids.extend(self.idx()?.ids_superseding(&cur.id).map_err(index_err)?);
        let mut seen = HashSet::new();
        Ok(ids
            .into_iter()
            .filter(|i| !i.is_empty() && seen.insert(i.clone()))
            .collect())
    }

    /// An unscoped load, for walking a chain. The caller applies the reader
    /// check to everything it keeps.
    fn raw_load(&self, id: &str) -> Option<Memory> {
        self.store
            .find_file(id)
            .and_then(|p| self.store.read_path(&p).ok())
    }
}

#[pymethods]
impl Engine {
    /// Open the store at `memory_dir` (the agent's `memory/` directory).
    ///
    /// `audience_config` is the path of the packaged `audience_config.json`
    /// (resolves per-call `channel:<id>` readers; it can only narrow, never
    /// widen the process ceiling). `fts=False` is the vaulted mode.
    #[new]
    #[pyo3(signature = (memory_dir, audience_config=None, fts=true))]
    fn new(memory_dir: PathBuf, audience_config: Option<PathBuf>, fts: bool) -> PyResult<Engine> {
        // A reader process never creates directories: only an owner process
        // may write to the store layout.
        let store = if OwnerScope::from_process().is_ok() {
            Store::open(&memory_dir).map_err(|e| PyOSError::new_err(e.to_string()))?
        } else {
            Store::open_readonly(&memory_dir)
        };
        let index = Index::open(&memory_dir, fts).map_err(index_err)?;
        let cfg = audience_config.map_or_else(AudienceConfig::empty, AudienceConfig::load);
        Ok(Engine {
            root: memory_dir,
            store,
            index: Mutex::new(index),
            cfg,
        })
    }

    /// The store root.
    #[getter]
    fn root(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    /// Search (`MemoryStore.search`): the lexical local arm fused by the Rust
    /// pipeline. Returns memory dicts (`model_dump` keys) plus `superseded` and
    /// `superseded_by`, or index rows when `detail="index"`.
    #[pyo3(signature = (query, limit=10, *, tags=None, layer=None, source=None, context_tag=None, record_access=false, reader=None, as_of=None, detail="full"))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        tags: Option<Vec<String>>,
        layer: Option<&str>,
        source: Option<&str>,
        context_tag: Option<&Bound<'_, PyAny>>,
        record_access: bool,
        reader: Option<&str>,
        as_of: Option<&str>,
        detail: &str,
    ) -> PyResult<PyObject> {
        if detail != "full" && detail != "index" {
            return Err(value_err("detail must be 'full' or 'index'"));
        }
        if let Some(l) = layer {
            parse_layer(l)?;
        }
        let as_of = match as_of {
            Some(a) => normalize_as_of(a).map_err(value_err)?,
            None => None,
        };
        let scope = self.scope(reader);
        let ctx = str_or_list(context_tag)?;
        let tags = tags.unwrap_or_default();
        let mems = self.run_search(
            query,
            limit,
            layer,
            &tags,
            source,
            ctx.as_deref(),
            as_of.as_deref(),
            &scope,
        )?;
        let mems: Vec<Memory> = mems.into_iter().take(limit).collect();
        if record_access {
            let ids: Vec<&str> = mems.iter().map(|m| m.id.as_str()).collect();
            self.idx()?
                .bump_access(&ids, None, Some(SEARCH_HIT_DEBOUNCE_S), &scope);
        }
        let sup = superseded_map(&mems, &mut |id| self.visible(id, &scope));
        let mut rows: Vec<Value> = Vec::new();
        for m in &mems {
            let mut v = m.to_value();
            let o = v.as_object_mut().expect("memory is an object");
            o.insert("superseded".into(), Value::Bool(sup.contains_key(&m.id)));
            o.insert(
                "superseded_by".into(),
                sup.get(&m.id)
                    .map_or(Value::Null, |s| Value::String(s.clone())),
            );
            rows.push(v);
        }
        if detail == "index" {
            let json_mod = py.import("json")?;
            let est = |v: &Value| -> PyResult<usize> {
                let kw = PyDict::new(py);
                kw.set_item("default", py.get_type::<pyo3::types::PyString>())?;
                let s = json_mod.call_method("dumps", (to_py(py, v)?,), Some(&kw))?;
                Ok(s.len()?.div_ceil(4))
            };
            let mut out = Vec::new();
            for (i, (m, full)) in mems.iter().zip(rows.iter()).enumerate() {
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
                let t = est(&row)?;
                let tf = est(full)?;
                let o = row.as_object_mut().expect("row is an object");
                o.insert("est_tokens".into(), t.into());
                o.insert("est_tokens_full".into(), tf.into());
                out.push(row);
            }
            return to_py(py, &Value::Array(out));
        }
        to_py(py, &Value::Array(rows))
    }

    /// `MemoryStore.recall`: the memory, or `None` when it does not exist or
    /// the reader may not see it (indistinguishable). A recall reinforces.
    #[pyo3(signature = (memory_id, reader=None))]
    fn recall(&self, py: Python<'_>, memory_id: &str, reader: Option<&str>) -> PyResult<PyObject> {
        let scope = self.scope(reader);
        let Some(mut m) = self.store.load(memory_id, &scope).map_err(store_err)? else {
            return Ok(py.None());
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
        self.idx()?.bump_access(&[memory_id], None, None, &scope);
        to_py(py, &m.to_value())
    }

    /// `MemoryStore.list_memories`: newest first, filtered, at most `limit`.
    #[pyo3(signature = (layer=None, tags=None, limit=50, created_after=None, created_before=None, source=None, reader=None))]
    #[allow(clippy::too_many_arguments)]
    fn list(
        &self,
        py: Python<'_>,
        layer: Option<&str>,
        tags: Option<Vec<String>>,
        limit: usize,
        created_after: Option<&str>,
        created_before: Option<&str>,
        source: Option<&str>,
        reader: Option<&str>,
    ) -> PyResult<PyObject> {
        let layer = layer.map(parse_layer).transpose()?;
        let scope = self.scope(reader);
        let tags = tags.unwrap_or_default();
        let listing = self.store.list(layer, &scope);
        let out: Vec<Value> = listing
            .memories
            .iter()
            .filter(|m| tags.iter().all(|t| m.tags.contains(t)))
            .filter(|m| source.is_none_or(|s| s.is_empty() || m.source == s))
            .filter(|m| created_after.is_none_or(|a| m.created_at.as_str() >= a))
            .filter(|m| created_before.is_none_or(|b| m.created_at.as_str() < b))
            .take(limit)
            .map(Memory::to_value)
            .collect();
        to_py(py, &Value::Array(out))
    }

    /// `MemoryStore.history`: the supersession chain through `memory_id`,
    /// oldest first. A hidden or unknown id is `[]`; the walk stops at the
    /// first neighbour the reader cannot see.
    #[pyo3(signature = (memory_id, reader=None))]
    fn history(&self, py: Python<'_>, memory_id: &str, reader: Option<&str>) -> PyResult<PyObject> {
        let scope = self.scope(reader);
        let seen_ok = |m: &Memory| scope.allows_memory(m);
        let Some(start) = self.raw_load(memory_id).filter(seen_ok) else {
            return to_py(py, &Value::Array(vec![]));
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
        to_py(py, &Value::Array(rows))
    }

    /// Whether `reader` may see `memory_id` (`MemoryStore.reader_can_see`).
    #[pyo3(signature = (memory_id, reader=None))]
    fn reader_can_see(&self, memory_id: &str, reader: Option<&str>) -> bool {
        self.visible(memory_id, &self.scope(reader))
    }

    /// Store-wide health. The owner gets the report; a declared reader gets
    /// only the verdict (the report carries counts of memories it may not see).
    #[pyo3(signature = (reader=None))]
    fn health(&self, py: Python<'_>, reader: Option<&str>) -> PyResult<PyObject> {
        let scope = self.scope(reader);
        let idx = self.idx()?;
        let fts_ok = !idx.fts_enabled() || idx.fts_ok();
        let owner_view = owner().ok().filter(|_| scope.unrestricted());
        let Some(own) = owner_view else {
            let v = json!({
                "engine": "skmem-rust",
                "ok": fts_ok,
                "status": if fts_ok { "ok" } else { "degraded" },
            });
            return to_py(py, &v);
        };
        let rs = own.reader_scope();
        let flat_ids = self.store.flat_id_set(rs);
        let sqlite_ids: BTreeSet<String> = idx
            .all_rows(rs)
            .map_err(index_err)?
            .into_iter()
            .map(|r| r.id)
            .collect();
        let sqlite_only = sqlite_ids.difference(&flat_ids).count();
        let flat_only = flat_ids.difference(&sqlite_ids).count();
        let unloadable = self.store.list(None, rs).unloadable.len();
        let wal = wal_status(idx.path(), WAL_DEGRADED_BYTES);
        let fts = idx.fts_status(&own).map_err(index_err)?;
        let in_sync = sqlite_only == 0 && flat_only == 0;
        let ok = unloadable == 0 && wal.wal_ok && fts_ok;
        let v = json!({
            "engine": "skmem-rust",
            "ok": ok,
            "status": if in_sync && ok { "ok" } else { "degraded" },
            "sqlite_only": sqlite_only,
            "flat_only": flat_only,
            "primary": {
                "ok": ok,
                "backend": "SkmemRust",
                "base_path": self.root.to_string_lossy(),
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
        });
        to_py(py, &v)
    }

    // ----- owner-only writes and maintenance -------------------------------

    /// Store a full memory dict (`model_dump` keys) exactly as given, then
    /// index it. Owner only. Returns the id. Use `snapshot` to build and seal one.
    #[pyo3(signature = (memory, seal=true))]
    fn save(&self, py: Python<'_>, memory: &Bound<'_, PyAny>, seal: bool) -> PyResult<String> {
        let own = owner()?;
        let mut m = Memory::from_stored(from_py(py, memory)?).map_err(value_err)?;
        if seal {
            m.seal();
        }
        self.store.save(&m, &own).map_err(store_err)?;
        let file = self.store.path_for(&m).map_err(store_err)?;
        self.idx()?
            .index_memory(&m, &file, &own)
            .map_err(index_err)?;
        Ok(m.id)
    }

    /// Create, seal, store and index a memory. Owner only. `fields` override
    /// any `Memory` field (`layer`, `tags`, `source`, `context_tag`,
    /// `source_tier`, ...). Returns the stored memory dict.
    #[pyo3(signature = (title, content, **fields))]
    fn snapshot(
        &self,
        py: Python<'_>,
        title: &str,
        content: &str,
        fields: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let own = owner()?;
        let mut base = Memory::new(title, content).to_value();
        if let Some(f) = fields {
            let extra = from_py(py, f.as_any())?;
            let o = base.as_object_mut().expect("memory is an object");
            for (k, v) in extra.as_object().cloned().unwrap_or_default() {
                o.insert(k, v);
            }
        }
        let mut m = Memory::from_stored(base).map_err(value_err)?;
        m.seal();
        self.store.save(&m, &own).map_err(store_err)?;
        let file = self.store.path_for(&m).map_err(store_err)?;
        self.idx()?
            .index_memory(&m, &file, &own)
            .map_err(index_err)?;
        to_py(py, &m.to_value())
    }

    /// Forget a memory: tombstone, remove the flat file and the index row.
    /// Owner only. Returns whether a memory existed.
    #[pyo3(signature = (memory_id, reason="forget", agent=None))]
    fn delete(&self, memory_id: &str, reason: &str, agent: Option<&str>) -> PyResult<bool> {
        let own = owner()?;
        if self.store.find_file(memory_id).is_none() {
            return Ok(false);
        }
        self.store
            .tombstone(memory_id, &forgotten_at(), reason, agent, &own)
            .map_err(store_err)?;
        let existed = self.store.delete(memory_id, &own).map_err(store_err)?;
        self.idx()?.delete(memory_id, &own).map_err(index_err)?;
        Ok(existed)
    }

    /// Move a memory to cold storage (`archive/`) and drop its index row.
    /// Owner only. Returns the new path or `None`.
    #[pyo3(signature = (memory_id, sub=None))]
    fn archive(&self, memory_id: &str, sub: Option<&str>) -> PyResult<Option<String>> {
        let own = owner()?;
        let moved = self
            .store
            .archive(memory_id, sub, &own)
            .map_err(store_err)?;
        if moved.is_some() {
            self.idx()?.delete(memory_id, &own).map_err(index_err)?;
        }
        Ok(moved.map(|p| p.to_string_lossy().into_owned()))
    }

    /// Rebuild the index from the flat files. Owner only.
    #[pyo3(signature = (force=false))]
    fn reindex(&self, py: Python<'_>, force: bool) -> PyResult<PyObject> {
        let own = owner()?;
        let r = self
            .idx()?
            .reindex(&self.store, force, &own)
            .map_err(index_err)?;
        to_py(
            py,
            &json!({
                "indexed": r.indexed, "unloadable": r.unloadable,
                "orphans_kept": r.orphans_kept, "usage_restored": r.usage_restored,
            }),
        )
    }

    // ----- owner-only intentions and ledger --------------------------------

    /// `memory_remind`: create a time intention. Owner only.
    #[pyo3(signature = (title, due_at, interval_s=None, origin_memory_id=None, channel=None))]
    fn remind(
        &self,
        py: Python<'_>,
        title: &str,
        due_at: &str,
        interval_s: Option<i64>,
        origin_memory_id: Option<&str>,
        channel: Option<&str>,
    ) -> PyResult<PyObject> {
        let own = owner()?;
        let title = flat(title);
        if title.is_empty() {
            return Err(value_err("title is required"));
        }
        if title.chars().count() > 500 {
            return Err(value_err("title is too long (500 characters max)"));
        }
        if interval_s.is_some_and(|s| s < 60) {
            return Err(value_err("interval_s must be at least 60 seconds"));
        }
        let due = parse_due_at(due_at, None).map_err(value_err)?;
        let id: String = crate::util::uuid4()
            .replace('-', "")
            .chars()
            .take(16)
            .collect();
        let mut i = Intention::new(&id, Kind::Time, &title);
        i.due_at = Some(due);
        i.interval_s = interval_s;
        i.origin_memory_id = origin_memory_id
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if let Some(c) = channel.filter(|c| !c.is_empty()) {
            i.channel = c.to_string();
        }
        let st = IntentionStore::open(&self.root, &own).map_err(value_err)?;
        st.save(&i).map_err(value_err)?;
        to_py(
            py,
            &json!({"id": i.id, "state": i.state.as_str(), "due_at": due.isoformat()}),
        )
    }

    /// `memory_intentions`: list intentions, optionally by state. Owner only.
    #[pyo3(signature = (state=None))]
    fn intentions(&self, py: Python<'_>, state: Option<&str>) -> PyResult<PyObject> {
        let own = owner()?;
        let st = IntentionStore::open(&self.root, &own).map_err(value_err)?;
        let rows: Vec<Value> = st
            .list(state)
            .map_err(value_err)?
            .iter()
            .map(|i| serde_json::from_str(&i.dump_compact()).unwrap_or(Value::Null))
            .collect();
        to_py(py, &Value::Array(rows))
    }

    /// `memory_intention_done`: apply an event (default `done`). Owner only.
    /// Returns `{"id", "state"}`, or `None` for an unknown id.
    #[pyo3(signature = (intention_id, event="done"))]
    fn intention_done(
        &self,
        py: Python<'_>,
        intention_id: &str,
        event: &str,
    ) -> PyResult<PyObject> {
        let own = owner()?;
        let st = IntentionStore::open(&self.root, &own).map_err(value_err)?;
        match st
            .apply(intention_id, event, None, None)
            .map_err(value_err)?
        {
            None => Ok(py.None()),
            Some(i) => to_py(py, &json!({"id": intention_id, "state": i.state.as_str()})),
        }
    }

    /// Append an outcome-ledger event. Owner only. Returns the event id.
    #[pyo3(signature = (skill, version, event, effect=None, session="", host=None))]
    fn ledger_record(
        &self,
        skill: &str,
        version: &str,
        event: &str,
        effect: Option<&str>,
        session: &str,
        host: Option<&str>,
    ) -> PyResult<String> {
        let own = owner()?;
        let led = Ledger::new(&self.root, host, &own);
        led.record(
            skill,
            version,
            event,
            &Record {
                effect,
                session,
                ..Record::default()
            },
        )
        .map_err(value_err)
    }

    /// Ledger scores, best first, at most `limit`. Owner only.
    #[pyo3(signature = (limit=20, host=None))]
    fn ledger_top(&self, py: Python<'_>, limit: usize, host: Option<&str>) -> PyResult<PyObject> {
        let own = owner()?;
        let led = Ledger::new(&self.root, host, &own);
        let rows: Vec<Value> = led.top(None, limit).iter().map(score_value).collect();
        to_py(py, &Value::Array(rows))
    }
}

/// Return the engine version.
#[pyfunction]
fn version() -> &'static str {
    super::version()
}

#[pymodule]
#[pyo3(name = "_skmem_core")]
fn skmem_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Engine>()?;
    m.add("__version__", super::version())?;
    Ok(())
}
