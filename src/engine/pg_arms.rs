//! Postgres adapters for the [`super::Engine`] arms (feature `pg`).
//!
//! The vector, BM25 and graph arms of `crate::pg` connect LAZILY here, on the
//! first search and inside the arm's worker thread, so connecting counts against
//! the search deadline and a dead server is a `failed` status per search (and is
//! retried on the next one) rather than an engine that cannot be opened.
//!
//! A read-only configuration adds `-c default_transaction_read_only=on` to the
//! connection options: every transaction on it is READ ONLY, so a bug cannot
//! write (`SET LOCAL` search settings are still allowed). The DSN is held only
//! as the connection string, never printed, never in an error.

use super::{Arms, Attach, Filters, GraphSource, MemoryArm};
use crate::audience::ReaderScope;
use crate::embed::{EmbedClient, Embedder};
use crate::model::{Layer, Memory};
use crate::pg::graph_arm::{graph_arm_enabled, GraphArm};
use crate::pg::lexical_arm::LexicalArm;
use crate::pg::{self, ContextTag, VectorArm};
use std::fmt;
use std::sync::Arc;

/// Builds the query embedder for each lazily created vector arm.
pub type EmbedderFactory = Arc<dyn Fn() -> Box<dyn Embedder + Send> + Send + Sync>;

/// Which Postgres arms to attach and how to reach them.
#[derive(Clone)]
pub struct PgConfig {
    dsn: String,
    /// Open every connection READ ONLY (`default_transaction_read_only=on`).
    pub read_only: bool,
    /// `SKMEMORY_PG_ALLOW_REMOTE`: allow a non-loopback host.
    pub allow_remote: bool,
    pub embedder: EmbedderFactory,
    /// Extra libpq-style connection options, for example `-c search_path=s,public`.
    pub options: Option<String>,
    pub vector: bool,
    pub lexical: bool,
    pub graph: bool,
}

impl fmt::Debug for PgConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PgConfig")
            .field("read_only", &self.read_only)
            .field("vector", &self.vector)
            .field("lexical", &self.lexical)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

impl PgConfig {
    /// All arms on, the HTTP embedder from `SKMEMORY_EMBED_URL` and friends.
    pub fn new(dsn: impl Into<String>, read_only: bool, allow_remote: bool) -> PgConfig {
        PgConfig {
            dsn: dsn.into(),
            read_only,
            allow_remote,
            embedder: Arc::new(|| Box::new(EmbedClient::from_env())),
            options: None,
            vector: true,
            lexical: true,
            graph: true,
        }
    }

    /// From `SKMEMORY_PG_DSN`; `None` when it is unset or blank (there is no
    /// implicit default DSN).
    pub fn from_env(read_only: bool) -> Option<PgConfig> {
        let dsn = std::env::var("SKMEMORY_PG_DSN").unwrap_or_default();
        if dsn.trim().is_empty() {
            return None;
        }
        let allow = matches!(
            std::env::var("SKMEMORY_PG_ALLOW_REMOTE")
                .unwrap_or_default()
                .trim()
                .to_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        );
        Some(PgConfig::new(dsn, read_only, allow))
    }

    /// The parsed connection settings (read-only option applied).
    pub fn connect_config(&self) -> Result<postgres::Config, String> {
        let mut c: postgres::Config = self
            .dsn
            .parse()
            .map_err(|_| "invalid connection string".to_string())?;
        let mut opts: Vec<String> = Vec::new();
        if let Some(o) = c.get_options().filter(|o| !o.is_empty()) {
            opts.push(o.to_string());
        }
        if let Some(o) = self.options.as_deref().filter(|o| !o.is_empty()) {
            opts.push(o.to_string());
        }
        if self.read_only {
            opts.push("-c default_transaction_read_only=on".to_string());
        }
        if !opts.is_empty() {
            c.options(&opts.join(" "));
        }
        Ok(c)
    }
}

struct Lazy<A> {
    arm: Option<A>,
    make: Box<dyn Fn() -> Result<A, String> + Send>,
}

impl<A> Lazy<A> {
    fn get(&mut self) -> Result<&mut A, String> {
        if self.arm.is_none() {
            self.arm = Some((self.make)()?);
        }
        Ok(self.arm.as_mut().expect("just set"))
    }
}

fn pg_filters(f: &Filters, with_context: bool) -> Result<pg::Filters, String> {
    let layer = match f.layer.as_deref() {
        None => None,
        Some(l) => Some(Layer::parse(l).ok_or_else(|| format!("unknown layer {l:?}"))?),
    };
    Ok(pg::Filters {
        layer,
        tags: f.tags.clone(),
        context_tag: if with_context {
            f.context_tags.clone().map(ContextTag::AnyOf)
        } else {
            None
        },
        source: f.source.clone(),
        as_of: f.as_of.clone(),
    })
}

struct Vector(Lazy<VectorArm>);

impl MemoryArm for Vector {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        f: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<Memory>, String> {
        let pf = pg_filters(f, true)?;
        let arm = self.0.get()?;
        arm.search_semantic(query, limit as i64, &pf, scope)
            .map(|hits| hits.into_iter().map(|h| h.memory).collect())
            .map_err(|e| e.to_string())
    }
}

/// `search_lexical` does not take `context_tag` in Python (the arm over-fetches
/// and the caller filters on the hydrated row), so none is pushed down here.
struct Lexical(Lazy<LexicalArm>);

impl MemoryArm for Lexical {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        f: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<Memory>, String> {
        let pf = pg_filters(f, false)?;
        let arm = self.0.get()?;
        arm.search_lexical(query, limit as i64, &pf, scope)
            .map(|hits| hits.into_iter().map(|h| h.memory).collect())
            .map_err(|e| e.to_string())
    }
}

struct Graph(GraphArm);

impl GraphSource for Graph {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        f: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<String>, String> {
        let pf = pg_filters(f, true)?;
        let out = self
            .0
            .search(query, limit, &pf, scope)
            .map_err(|e| e.to_string())?;
        Ok(out.hits.into_iter().filter_map(|h| h.id).collect())
    }
}

/// The vector arm for `agent` (connects on first use).
pub fn vector_arm(cfg: &PgConfig, agent: &str) -> Box<dyn MemoryArm> {
    let (cfg, agent) = (cfg.clone(), agent.to_string());
    Box::new(Vector(Lazy {
        arm: None,
        make: Box::new(move || {
            let config = cfg.connect_config()?;
            VectorArm::connect_config(config, &agent, (cfg.embedder)(), cfg.allow_remote)
                .map_err(|e| e.to_string())
        }),
    }))
}

/// The BM25 lexical arm for `agent` (connects on first use).
pub fn lexical_arm(cfg: &PgConfig, agent: &str) -> Box<dyn MemoryArm> {
    let (cfg, agent) = (cfg.clone(), agent.to_string());
    Box::new(Lexical(Lazy {
        arm: None,
        make: Box::new(move || {
            let config = cfg.connect_config()?;
            LexicalArm::connect_config(config, &agent, cfg.allow_remote).map_err(|e| e.to_string())
        }),
    }))
}

/// Attach the arms `cfg` asks for, scoped to `agent`. The graph arm is OFF
/// unless `SKMEMORY_GRAPH_ARM` turns it on (the reason is carried as the
/// `disabled` detail), exactly as in Python.
pub fn attach(cfg: &PgConfig, agent: &str) -> Arms {
    let semantic = if cfg.vector {
        Attach::ready(vector_arm(cfg, agent))
    } else {
        Attach::Absent
    };
    let lexical = if cfg.lexical {
        Attach::ready(lexical_arm(cfg, agent))
    } else {
        Attach::Absent
    };
    let graph = if !cfg.graph {
        Attach::Absent
    } else {
        match cfg.connect_config().and_then(|c| {
            GraphArm::new(
                c,
                agent,
                None,
                graph_arm_enabled(|k| std::env::var(k).ok()),
                cfg.allow_remote,
            )
            .map_err(|e| e.to_string())
        }) {
            Err(e) => Attach::Failed(e),
            Ok(arm) => {
                let st = arm.state();
                match st.state {
                    "enabled" => Attach::ready(Box::new(Graph(arm)) as Box<dyn GraphSource>),
                    "disabled" => Attach::Disabled(st.reason.unwrap_or_default()),
                    _ => Attach::Failed(st.reason.unwrap_or_else(|| "unavailable".into())),
                }
            }
        }
    };
    Arms {
        semantic,
        lexical,
        graph,
    }
}
