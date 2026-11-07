//! The store engine with the full arm set (SKMEM-R-07C): the Rust twin of
//! `MemoryStore.search` -> `search_fused` over the five arms.
//!
//! * `semantic`: dense cosine over the vector backend (Postgres pgvector, or any
//!   [`MemoryArm`]),
//! * `lexical`: Postgres BM25 when a lexical backend is attached, else the
//!   SQLite `LIKE` search (`SQLiteBackend.search_text`), as Python does,
//! * `lexical_local`: SQLite FTS5 bm25,
//! * `graph`: one-hop entity expansion (OFF unless `SKMEMORY_GRAPH_ARM`),
//! * `temporal`: a date-bounded listing, only when the query names a time window.
//!
//! The pre-fusion arms run concurrently under ONE total deadline
//! (`SKMEMORY_SEARCH_DEADLINE_S`, default 4 s): an arm still running at the
//! deadline is reported `timeout`, never as an empty result, and its thread is
//! abandoned to finish (its result is dropped). A second search never queues
//! behind an abandoned call: an arm whose previous call is still running
//! reports `timeout` at once (Python's `ArmGuard`). After the arms the pipeline
//! in [`crate::search`] runs unchanged (route, vector pin, interleave or RRF,
//! rare-term rerank, supersession demotion, decay, tier cap, procedure order).
//!
//! Scope guarantees (R-06C, R-06D, R-07B) are kept: every arm receives the
//! reader scope and filters in its own query, hydrated rows are re-checked, a
//! restricted reader's SQLite handle is read-only, and [`crate::search::search`]
//! drops anything the reader may not see before fusing.

use crate::audience::{AudienceConfig, OwnerScope, ReaderScope};
use crate::decay::{self, Usage};
use crate::env::Env;
use crate::flat::{Store, StoreError};
use crate::fusion::{run_arm, ArmInput, ArmOutcome, ArmStatus, Route, DEFAULT_ARM_CAP};
use crate::index::{Index, IndexError, SearchFilters};
use crate::model::{Layer, Memory};
use crate::procedures::order_procedures;
use crate::pytime::Dt;
use crate::query_sanitizer::sanitize_query;
use crate::search::{demote_superseded, search, Candidate, SearchInput};
use crate::tiers::apply_tier_cap;
use crate::validity::valid_at;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

#[cfg(feature = "pg")]
pub mod pg_arms;

/// `SKMEMORY_SEARCH_DEADLINE_S`.
pub const DEADLINE_ENV: &str = "SKMEMORY_SEARCH_DEADLINE_S";
/// `DEFAULT_DEADLINE_S`.
pub const DEFAULT_DEADLINE_S: f64 = 4.0;
/// `MemoryStore.last_search_degraded()` when an arm missed the deadline.
pub const DEGRADED_ARMS_TIMED_OUT: &str = "arms timed out";
/// The arms, in the order Python lists them.
pub const ARM_NAMES: [&str; 5] = ["semantic", "lexical", "lexical_local", "graph", "temporal"];

/// `search_deadline_s`: a positive number of seconds, else the default.
pub fn search_deadline(env: &Env) -> Duration {
    let secs = env
        .get(DEADLINE_ENV)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0 && v.is_finite())
        .unwrap_or(DEFAULT_DEADLINE_S);
    Duration::from_secs_f64(secs.min(86_400.0))
}

/// What can go wrong outside an arm (an arm's own failure is a status).
#[derive(Debug)]
pub enum EngineError {
    Store(StoreError),
    Index(IndexError),
    Io(std::io::Error),
    /// A write on a read-only engine.
    ReadOnly,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Store(e) => write!(f, "{e}"),
            EngineError::Index(e) => write!(f, "{e}"),
            EngineError::Io(e) => write!(f, "io: {e}"),
            EngineError::ReadOnly => write!(f, "this engine is read-only"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<StoreError> for EngineError {
    fn from(e: StoreError) -> Self {
        EngineError::Store(e)
    }
}

impl From<IndexError> for EngineError {
    fn from(e: IndexError) -> Self {
        EngineError::Index(e)
    }
}

/// The filters every arm receives (already validated; `as_of` is the
/// normalised UTC instant from [`crate::validity::normalize_as_of`]).
#[derive(Clone, Debug, Default)]
pub struct Filters {
    pub layer: Option<String>,
    /// AND logic.
    pub tags: Vec<String>,
    pub source: Option<String>,
    /// Exact `context_tag` values, any-of. `Some(vec![])` matches nothing.
    pub context_tags: Option<Vec<String>>,
    pub as_of: Option<String>,
}

/// A vector or lexical backend: returns hydrated memories, best first.
/// `Err` is the arm's failure message (it must never contain a secret).
pub trait MemoryArm: Send {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<Memory>, String>;
}

/// A graph backend: returns memory ids, best first.
pub trait GraphSource: Send {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<String>, String>;
}

/// Shared, lockable arm handle.
pub type Shared<T> = Arc<Mutex<Box<T>>>;

/// How an arm is attached.
pub enum Attach<T: ?Sized> {
    /// No backend (`unavailable`: "backend not attached").
    Absent,
    /// Switched off by configuration, with the reason (`disabled`).
    Disabled(String),
    /// Could not be built (`failed`, with the reason).
    Failed(String),
    Ready(Shared<T>),
}

impl<T: ?Sized> Attach<T> {
    /// Wrap a backend.
    pub fn ready(arm: Box<T>) -> Attach<T> {
        Attach::Ready(Arc::new(Mutex::new(arm)))
    }

    fn attached(&self) -> bool {
        !matches!(self, Attach::Absent)
    }
}

/// The pre-fusion backends of an [`Engine`].
pub struct Arms {
    pub semantic: Attach<dyn MemoryArm>,
    /// A lexical BM25 backend. When [`Attach::Absent`], the `lexical` arm is
    /// the SQLite `LIKE` search, as in Python without Postgres.
    pub lexical: Attach<dyn MemoryArm>,
    pub graph: Attach<dyn GraphSource>,
}

impl Arms {
    /// Nothing attached: SQLite only (`lexical` is `LIKE`, the rest abstain).
    pub fn none() -> Arms {
        Arms {
            semantic: Attach::Absent,
            lexical: Attach::Absent,
            graph: Attach::Absent,
        }
    }
}

/// How the engine is opened.
pub struct EngineOptions {
    pub audience_config: Option<PathBuf>,
    /// FTS on (`false` is the vaulted mode).
    pub fts: bool,
    /// `Some(true)` or `Some(false)` forces it; `None` is read-only exactly
    /// when the process is not the owner (a declared `SKMEMORY_READER`).
    pub read_only: Option<bool>,
    pub arms: Arms,
}

impl EngineOptions {
    pub fn new(arms: Arms) -> EngineOptions {
        EngineOptions {
            audience_config: None,
            fts: true,
            read_only: None,
            arms,
        }
    }
}

/// One search.
#[derive(Clone, Debug, Default)]
pub struct SearchRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub layer: Option<&'a str>,
    pub tags: &'a [String],
    pub source: Option<&'a str>,
    /// `None` is no filter; `Some(&[])` matches nothing.
    pub context_tags: Option<&'a [String]>,
    /// Already normalised (see [`crate::validity::normalize_as_of`]).
    pub as_of: Option<&'a str>,
    /// The clock the temporal arm and decay read; `None` is the wall clock.
    /// Tests and the bench pin it; the Python binding never sets it.
    pub now: Option<Dt>,
}

/// Which code path produced the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPath {
    /// The fused arms.
    Fused,
    /// No arm answered and none timed out: the legacy SQLite `LIKE` path.
    LegacyLike,
    /// No arm answered and at least one timed out: empty, flagged degraded.
    Empty,
}

/// What a search did, for the retrieval log and `arm_diff`. Never the query.
#[derive(Clone, Debug)]
pub struct SearchReport {
    pub arms: Vec<ArmOutcome>,
    pub route: Option<Route>,
    pub rerank_applied: bool,
    /// `Some(DEGRADED_ARMS_TIMED_OUT)` when an arm missed the deadline.
    pub degraded: Option<&'static str>,
    pub elapsed: Duration,
    pub deadline: Duration,
    pub path: SearchPath,
    /// The fused id order after each stage (empty off the fused path).
    pub fused: Vec<String>,
}

/// The page and the report.
#[derive(Debug)]
pub struct SearchOutput {
    pub memories: Vec<Memory>,
    pub report: SearchReport,
}

enum Payload {
    Mems(Vec<Memory>),
    Ids(Vec<String>),
}

enum ArmRun {
    Done(Result<Payload, String>),
    /// The previous call on this arm is still running (an abandoned thread).
    Busy,
}

/// Run `f` on the slot unless a previous call still holds it.
fn guarded<T: ?Sized, R>(
    slot: &Shared<T>,
    f: impl FnOnce(&mut T) -> Result<R, String>,
) -> Result<R, Option<String>> {
    match slot.try_lock() {
        Ok(mut g) => f(&mut g).map_err(Some),
        Err(TryLockError::WouldBlock) => Err(None),
        Err(TryLockError::Poisoned(p)) => f(&mut p.into_inner()).map_err(Some),
    }
}

/// A store rooted at an agent's `memory/` directory: flat files, SQLite index
/// and the pre-fusion arms.
pub struct Engine {
    root: PathBuf,
    store: Store,
    index: Mutex<Index>,
    cfg: AudienceConfig,
    arms: Arms,
    read_only: bool,
}

impl Engine {
    /// Open the store at `memory_dir`. A read-only engine never creates a
    /// directory or a file and never writes `index.db` (see
    /// [`Index::open_readonly`]).
    pub fn open(
        memory_dir: impl Into<PathBuf>,
        opts: EngineOptions,
    ) -> Result<Engine, EngineError> {
        let root: PathBuf = memory_dir.into();
        let read_only = opts
            .read_only
            .unwrap_or_else(|| OwnerScope::from_process().is_err());
        let store = if read_only {
            Store::open_readonly(&root)
        } else {
            Store::open(&root).map_err(EngineError::Io)?
        };
        let index = if read_only {
            Index::open_readonly(&root, opts.fts)?
        } else {
            Index::open(&root, opts.fts)?
        };
        let cfg = opts
            .audience_config
            .map_or_else(AudienceConfig::empty, AudienceConfig::load);
        Ok(Engine {
            root,
            store,
            index: Mutex::new(index),
            cfg,
            arms: opts.arms,
            read_only,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Whether this engine never writes.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// The index handle. Poison is recovered: the index holds no invariant a
    /// panicking reader could break.
    pub fn index(&self) -> std::sync::MutexGuard<'_, Index> {
        self.index.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The reader scope for one call: the per-call `reader` resolved under this
    /// engine's audience config, the process ceiling under the process one.
    pub fn scope(&self, reader: Option<&str>) -> ReaderScope {
        ReaderScope::from_process_with_config(reader, &self.cfg)
    }

    /// Which arms are attached, by name (`Absent` arms are `false`).
    pub fn attached(&self) -> [(&'static str, bool); 3] {
        [
            ("semantic", self.arms.semantic.attached()),
            ("lexical", self.arms.lexical.attached()),
            ("graph", self.arms.graph.attached()),
        ]
    }

    /// Owner-only: save and index a memory. Refused when read-only.
    pub fn save(&self, m: &Memory, owner: &OwnerScope) -> Result<String, EngineError> {
        if self.read_only {
            return Err(EngineError::ReadOnly);
        }
        self.store.save(m, owner)?;
        let file = self.store.path_for(m)?;
        self.index().index_memory(m, &file, owner)?;
        Ok(m.id.clone())
    }

    fn accept(m: &Memory, f: &Filters, check_ctx: bool) -> bool {
        if check_ctx {
            if let Some(ctx) = &f.context_tags {
                if !ctx.contains(&m.context_tag) {
                    return false;
                }
            }
        }
        f.layer.as_deref().is_none_or(|l| m.layer.as_str() == l)
            && f.tags.iter().all(|t| m.tags.contains(t))
            && f.source
                .as_deref()
                .is_none_or(|s| s.is_empty() || m.source == s)
            && f.as_of.as_deref().is_none_or(|a| {
                valid_at(
                    m.valid_from.as_deref(),
                    Some(&m.created_at),
                    m.valid_to.as_deref(),
                    a,
                )
            })
    }

    /// `MemoryStore.search` over every attached arm. Returns the page (at most
    /// `req.limit`) and the report.
    pub fn search(
        &self,
        req: &SearchRequest<'_>,
        scope: &ReaderScope,
        env: &Env,
    ) -> Result<SearchOutput, EngineError> {
        let started = Instant::now();
        let deadline = search_deadline(env);
        let now = req.now.unwrap_or_else(Dt::now);
        let query = sanitize_query(req.query);
        let limit = req.limit;
        let pool = limit.max(limit * 3);
        let arm_limit = (pool * 3).max(DEFAULT_ARM_CAP);
        let filters = Filters {
            layer: req.layer.map(str::to_string),
            tags: req.tags.to_vec(),
            source: req.source.map(str::to_string),
            context_tags: req.context_tags.map(<[String]>::to_vec),
            as_of: req.as_of.filter(|a| !a.is_empty()).map(str::to_string),
        };

        // ---- pg arms: one thread each, results over a channel ----------------
        let (tx, rx) = mpsc::channel::<(usize, ArmRun)>();
        let mut pending = 0usize;
        let mut prelim: [Option<ArmInput>; 5] = [None, None, None, None, None];
        let mut payloads: [Option<Payload>; 5] = [None, None, None, None, None];
        let mut spawn_mem =
            |i: usize, attach: &Attach<dyn MemoryArm>, pending: &mut usize| match attach {
                Attach::Absent => {
                    if i == 0 {
                        prelim[i] = Some(ArmInput::Unattached);
                    }
                }
                Attach::Disabled(why) => prelim[i] = Some(ArmInput::Disabled(why.clone())),
                Attach::Failed(why) => prelim[i] = Some(ArmInput::Failed(why.clone())),
                Attach::Ready(slot) => {
                    let (slot, tx) = (Arc::clone(slot), tx.clone());
                    let (q, f, sc) = (query.clone(), filters.clone(), scope.clone());
                    let spawned =
                        std::thread::Builder::new()
                            .name("skmem-arm".into())
                            .spawn(move || {
                                let run = match guarded(&slot, |a| a.search(&q, arm_limit, &f, &sc))
                                {
                                    Ok(m) => ArmRun::Done(Ok(Payload::Mems(m))),
                                    Err(Some(e)) => ArmRun::Done(Err(e)),
                                    Err(None) => ArmRun::Busy,
                                };
                                let _ = tx.send((i, run));
                            });
                    match spawned {
                        Ok(_) => *pending += 1,
                        Err(e) => prelim[i] = Some(ArmInput::Failed(format!("spawn: {e}"))),
                    }
                }
            };
        spawn_mem(0, &self.arms.semantic, &mut pending);
        spawn_mem(1, &self.arms.lexical, &mut pending);
        match &self.arms.graph {
            Attach::Absent => prelim[3] = Some(ArmInput::Unattached),
            Attach::Disabled(why) => prelim[3] = Some(ArmInput::Disabled(why.clone())),
            Attach::Failed(why) => prelim[3] = Some(ArmInput::Failed(why.clone())),
            Attach::Ready(slot) => {
                let (slot, tx) = (Arc::clone(slot), tx.clone());
                let (q, f, sc) = (query.clone(), filters.clone(), scope.clone());
                let spawned =
                    std::thread::Builder::new()
                        .name("skmem-arm".into())
                        .spawn(move || {
                            let run = match guarded(&slot, |a| a.search(&q, arm_limit, &f, &sc)) {
                                Ok(ids) => ArmRun::Done(Ok(Payload::Ids(ids))),
                                Err(Some(e)) => ArmRun::Done(Err(e)),
                                Err(None) => ArmRun::Busy,
                            };
                            let _ = tx.send((3, run));
                        });
                match spawned {
                    Ok(_) => pending += 1,
                    Err(e) => prelim[3] = Some(ArmInput::Failed(format!("spawn: {e}"))),
                }
            }
        }
        drop(tx);

        // ---- local arms (SQLite), on this thread while the pg arms run --------
        let sf = SearchFilters {
            layer: filters.layer.as_deref(),
            tags: &filters.tags,
            source: filters.source.as_deref(),
            context_tags: filters.context_tags.as_deref(),
            as_of: filters.as_of.as_deref(),
        };
        let mut found: HashMap<String, Memory> = HashMap::new();
        let load = |id: &str| -> Option<Memory> { self.store.load(id, scope).ok().flatten() };
        let idx = self.index();
        // lexical_local
        let local_ids: Vec<String> = match idx.search_lexical(&query, arm_limit, &sf, scope) {
            Ok(rows) => rows.into_iter().map(|r| r.id).collect(),
            Err(e) => {
                prelim[2] = Some(ArmInput::Failed(e.to_string()));
                Vec::new()
            }
        };
        // lexical without a BM25 backend: SQLite LIKE
        let like_ids: Option<Vec<String>> = if matches!(self.arms.lexical, Attach::Absent) {
            match idx.search_text_like(&query, arm_limit, scope) {
                Ok(v) => Some(v),
                Err(e) => {
                    prelim[1] = Some(ArmInput::Failed(e.to_string()));
                    None
                }
            }
        } else {
            None
        };
        // temporal
        let temporal_ids: Vec<String> =
            match crate::bench::time_window(&query, crate::bench::unix_secs(now)) {
                None => Vec::new(),
                Some((after, before)) => {
                    match idx.created_between(&after, &before, &sf, arm_limit, scope) {
                        Ok(v) => v,
                        Err(e) => {
                            prelim[4] = Some(ArmInput::Failed(e.to_string()));
                            Vec::new()
                        }
                    }
                }
            };

        // ---- wait for the pg arms under the one deadline ----------------------
        let mut reached = [false; 5];
        while pending > 0 {
            let left = deadline.saturating_sub(started.elapsed());
            match rx.recv_timeout(left) {
                Ok((i, run)) => {
                    pending -= 1;
                    reached[i] = true;
                    match run {
                        ArmRun::Done(Ok(p)) => payloads[i] = Some(p),
                        ArmRun::Done(Err(e)) => prelim[i] = Some(ArmInput::Failed(e)),
                        ArmRun::Busy => {
                            prelim[i] = Some(ArmInput::TimedOut(
                                "previous call on this arm is still running".into(),
                            ))
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let elapsed_now = started.elapsed();
        for i in [0usize, 1, 3] {
            let spawned_arm = match i {
                0 => matches!(self.arms.semantic, Attach::Ready(_)),
                1 => matches!(self.arms.lexical, Attach::Ready(_)),
                _ => matches!(self.arms.graph, Attach::Ready(_)),
            };
            if spawned_arm && !reached[i] && prelim[i].is_none() {
                prelim[i] = Some(if elapsed_now >= deadline {
                    ArmInput::TimedOut(format!(
                        "no answer within {:.2}s deadline ({:.2}s elapsed)",
                        deadline.as_secs_f64(),
                        elapsed_now.as_secs_f64()
                    ))
                } else {
                    ArmInput::Failed("arm thread ended without an answer".into())
                });
            }
        }

        // ---- classify and hydrate, arm by arm in Python's order ---------------
        let mut ids_for: [Option<Vec<String>>; 5] = [None, None, None, None, None];
        let take_mems = |ms: Vec<Memory>, check_ctx: bool, found: &mut HashMap<String, Memory>| {
            let mut ids = Vec::new();
            for m in ms {
                if m.id.is_empty() || !Self::accept(&m, &filters, check_ctx) {
                    continue;
                }
                ids.push(m.id.clone());
                found.entry(m.id.clone()).or_insert(m);
            }
            ids
        };
        let take_ids = |ids: Vec<String>, check_ctx: bool, found: &mut HashMap<String, Memory>| {
            let mut out = Vec::new();
            for id in ids {
                if let Some(m) = found.get(&id) {
                    if Self::accept(m, &filters, check_ctx) {
                        out.push(id);
                    }
                    continue;
                }
                let Some(m) = load(&id) else { continue };
                if Self::accept(&m, &filters, check_ctx) {
                    out.push(id.clone());
                    found.insert(id, m);
                }
            }
            out
        };
        for i in 0..5 {
            if prelim[i].is_some() {
                continue;
            }
            ids_for[i] = Some(match i {
                0 => match payloads[0].take() {
                    Some(Payload::Mems(ms)) => take_mems(ms, false, &mut found),
                    _ => Vec::new(),
                },
                1 => match (payloads[1].take(), &like_ids) {
                    (Some(Payload::Mems(ms)), _) => take_mems(ms, true, &mut found),
                    (_, Some(ids)) => take_ids(ids.clone(), true, &mut found),
                    _ => Vec::new(),
                },
                2 => take_ids(local_ids.clone(), false, &mut found),
                3 => match payloads[3].take() {
                    Some(Payload::Ids(ids)) => take_ids(ids, false, &mut found),
                    _ => Vec::new(),
                },
                _ => take_ids(temporal_ids.clone(), true, &mut found),
            });
        }
        let cap = DEFAULT_ARM_CAP;
        let arms: Vec<ArmOutcome> = (0..5)
            .map(|i| {
                let input = match (prelim[i].take(), ids_for[i].take()) {
                    (Some(p), _) => p,
                    (None, Some(ids)) => ArmInput::Ids(ids),
                    (None, None) => ArmInput::Unattached,
                };
                run_arm(ARM_NAMES[i], input, cap)
            })
            .collect();
        let timed_out = arms.iter().any(|a| a.status == ArmStatus::Timeout);

        // ---- fuse, rerank, demote, decay, cap, order --------------------------
        let candidates: HashMap<String, Candidate> = found
            .iter()
            .map(|(k, m)| (k.clone(), Candidate::from_memory(m)))
            .collect();
        let all_ids: Vec<String> = found.keys().cloned().collect();
        let usage: HashMap<String, Usage> = {
            let refs: Vec<&str> = all_ids.iter().map(String::as_str).collect();
            idx.get_usage(&refs, scope)
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
        let df = |t: &str| idx.term_doc_freq(t);
        let input = SearchInput {
            query: &query,
            scope,
            limit,
            arms: &arms,
            found: &candidates,
            has_pg_bm25: self.arms.lexical.attached(),
            df: Some(&df),
            usage: Some(&usage),
            decay_as_of: now,
        };
        let trace = search(&input, env);
        let mut report = SearchReport {
            arms: arms.clone(),
            route: None,
            rerank_applied: false,
            degraded: timed_out.then_some(DEGRADED_ARMS_TIMED_OUT),
            elapsed: started.elapsed(),
            deadline,
            path: SearchPath::Fused,
            fused: Vec::new(),
        };
        if let Some(t) = trace {
            report.route = t.route;
            report.rerank_applied = t.rerank_applied;
            report.fused = t.fused;
            let memories = t
                .page
                .into_iter()
                .filter_map(|i| found.remove(&i))
                .collect();
            report.elapsed = started.elapsed();
            return Ok(SearchOutput { memories, report });
        }

        // ---- nothing fused ------------------------------------------------------
        if timed_out {
            // Never the legacy path: it would call the hung backends again.
            report.path = SearchPath::Empty;
            report.elapsed = started.elapsed();
            return Ok(SearchOutput {
                memories: Vec::new(),
                report,
            });
        }
        // The legacy path: Python tries the vector backend's plain text search
        // (not ported, see docs), the local FTS (it just ran in the lexical_local
        // arm), then SQLite LIKE. Only LIKE can add anything here.
        report.path = SearchPath::LegacyLike;
        let fetch = if filters.context_tags.is_some() || filters.as_of.is_some() {
            pool * 3
        } else {
            pool
        };
        let like = idx.search_text_like(&query, fetch, scope)?;
        let mut kept: Vec<Memory> = Vec::new();
        for id in like {
            let Some(m) = load(&id) else { continue };
            if Self::accept(&m, &filters, true) {
                kept.push(m);
            }
            if kept.len() >= pool {
                break;
            }
        }
        let cands: HashMap<String, Candidate> = kept
            .iter()
            .map(|m| (m.id.clone(), Candidate::from_memory(m)))
            .collect();
        let ids: Vec<String> = kept.iter().map(|m| m.id.clone()).collect();
        let usage_legacy: HashMap<String, Usage> = {
            let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            idx.get_usage(&refs, scope)
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
        let after_decay = match decay::configured_channel(env) {
            Some(ch) if ids.len() >= 2 => decay::rerank(
                &ids,
                &cands,
                Some(&usage_legacy),
                ch,
                now,
                decay::RERANK_GAMMA,
            ),
            _ => ids.clone(),
        };
        let demoted = demote_superseded(&after_decay, &cands);
        let capped = apply_tier_cap(&demoted, &cands, limit, env);
        let page = order_procedures(&capped, &cands, &query);
        let mut by_id: HashMap<String, Memory> =
            kept.into_iter().map(|m| (m.id.clone(), m)).collect();
        let memories = page.into_iter().filter_map(|i| by_id.remove(&i)).collect();
        report.elapsed = started.elapsed();
        Ok(SearchOutput { memories, report })
    }

    /// Layer filter helper for callers holding a string.
    pub fn parse_layer(s: &str) -> Option<Layer> {
        Layer::parse(s)
    }
}
