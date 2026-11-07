//! The pgvector search arm, the Rust twin of
//! `PGVectorBackend.search_semantic` (SKMEM-R-04). Feature `pg`.
//!
//! Dense cosine search over `public.memories`, scoped to one agent, with
//! every filter (layer, tags AND, context_tag, source, as_of) in the SQL
//! `WHERE` so `LIMIT` truncates only rows that already match. The index scan
//! sits in a materialised CTE and the outer query re-sorts by the real
//! distance, because `hnsw.iterative_scan = relaxed_order` can return rows
//! slightly out of order. Every value is a bound parameter.
//!
//! The DSN is read once and handed to the driver: it is never stored on the
//! arm, never put in an error, and never in `Debug` output. No TLS (`NoTls`):
//! skmem-pg is per-node on loopback.

pub mod graph_arm;
pub mod lexical_arm;

use crate::embed::{EmbedError, Embedder, VECTOR_DIM};
use crate::model::{Layer, Memory};
use pgvector::Vector;
use postgres::types::ToSql;
use postgres::{Client, NoTls};
use serde_json::Value;
use std::fmt;

/// `PGVectorBackend`'s default `hnsw.ef_search` (pgvector's own is 40).
pub const DEFAULT_HNSW_EF_SEARCH: i64 = 200;
/// Default `hnsw.iterative_scan`.
pub const DEFAULT_HNSW_ITERATIVE_SCAN: &str = "relaxed_order";

const V2_COLUMNS: [&str; 8] = [
    "kind",
    "outcome",
    "event_at",
    "valid_from",
    "valid_to",
    "supersedes_id",
    "source_tier",
    "schema_version",
];

/// Errors from the arm. None of them can contain the DSN.
#[derive(Debug)]
pub enum PgError {
    /// `SKMEMORY_PG_DSN` is unset or empty (there is no implicit default).
    NoDsn,
    /// Could not connect; the message has the DSN scrubbed.
    Connect(String),
    Query(String),
    Embed(EmbedError),
    /// The query vector is not `expected` long (a wrong-model embedding).
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    /// `as_of` is not an ISO-8601 timestamp (checked in Rust before any SQL).
    BadAsOf(String),
    /// The DSN targets a non-loopback host and this connection has no TLS.
    /// Set `SKMEMORY_PG_ALLOW_REMOTE=1` (or use [`VectorArm::connect_with`]
    /// with `allow_remote`) for a deliberate, audited exception.
    RemoteDsn {
        host: String,
    },
    /// The connection's effective `search_path` names no existing schema, so
    /// an unqualified `memories` cannot resolve. Postgres says only
    /// `relation "memories" does not exist`; this says why.
    EmptySearchPath,
    /// The graph name is not a plain identifier (it is the one piece of text
    /// AGE requires inline in the query, so it is validated, never bound).
    BadGraphName(String),
    /// A statement hit its `statement_timeout` (SQLSTATE 57014). The graph
    /// search arm treats this as abstaining, not as a failure.
    Timeout(String),
}

impl fmt::Display for PgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PgError::NoDsn => write!(f, "SKMEMORY_PG_DSN is not set"),
            PgError::Connect(m) => write!(f, "postgres connect failed: {m}"),
            PgError::Query(m) => write!(f, "postgres query failed: {m}"),
            PgError::Embed(e) => write!(f, "{e}"),
            PgError::DimensionMismatch { expected, actual } => write!(
                f,
                "query vector has {actual} dimensions, expected {expected}"
            ),
            PgError::BadAsOf(v) => write!(f, "as_of is not an ISO-8601 timestamp: {v:?}"),
            PgError::EmptySearchPath => write!(
                f,
                "the connection's search_path resolves to no existing schema (empty or all names missing): set search_path to the schema that holds memories, for example `public`"
            ),
            PgError::Timeout(m) => write!(f, "postgres statement timed out: {m}"),
            PgError::BadGraphName(n) => write!(
                f,
                "refusing graph name {n:?}: it must match [A-Za-z_][A-Za-z0-9_]*"
            ),
            PgError::RemoteDsn { host } => write!(
                f,
                "refusing to connect to non-loopback host {host:?}: skmem-pg is per-node and this client does not use TLS, so the password would cross the network in plaintext; point SKMEMORY_PG_DSN at localhost or set SKMEMORY_PG_ALLOW_REMOTE=1 for an audited exception"
            ),
        }
    }
}

impl std::error::Error for PgError {}

impl From<EmbedError> for PgError {
    fn from(e: EmbedError) -> Self {
        match e {
            EmbedError::DimensionMismatch {
                expected, actual, ..
            } => PgError::DimensionMismatch { expected, actual },
            other => PgError::Embed(other),
        }
    }
}

/// `context_tag` filter: one exact tag, or any-of a list. An empty list
/// matches nothing (fail closed), never everything.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextTag {
    One(String),
    AnyOf(Vec<String>),
}

impl From<crate::validity::BadAsOf> for PgError {
    fn from(e: crate::validity::BadAsOf) -> Self {
        PgError::BadAsOf(e.0)
    }
}

/// Search filters, all optional. Empty `tags`, empty `source` and empty
/// `as_of` mean "no filter", as in Python's truthiness checks.
#[derive(Clone, Debug, Default)]
pub struct Filters {
    pub layer: Option<Layer>,
    /// AND logic: the row must carry every tag.
    pub tags: Vec<String>,
    pub context_tag: Option<ContextTag>,
    pub source: Option<String>,
    /// ISO-8601 instant, parsed in Rust (see [`crate::validity`]); keeps rows
    /// valid then (see [`as_of_clause`]). `now` or `yesterday` are errors.
    pub as_of: Option<String>,
}

/// One result row.
#[derive(Debug)]
pub struct Hit {
    pub memory: Memory,
    /// Cosine distance (`<=>`), ascending.
    pub distance: f64,
}

/// The pgvector arm over one agent.
pub struct VectorArm {
    client: Client,
    agent: String,
    embedder: Box<dyn Embedder + Send>,
    v2_columns: Option<bool>,
    /// Connection settings kept so a dead connection can be re-established.
    /// Never printed (see the `Debug` impl); absent for `from_client`.
    reconnect: Option<postgres::Config>,
    /// Rows this arm could not parse (skipped, as Python does).
    pub unparseable: u64,
}

impl fmt::Debug for VectorArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorArm")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

/// Remove each non-empty secret (the password) from an error message.
pub(crate) fn scrub(msg: &str, secrets: &[&str]) -> String {
    let mut out = msg.to_string();
    for s in secrets.iter().filter(|s| !s.is_empty()) {
        out = out.replace(s, "<redacted>");
    }
    out
}

/// `_check_dsn_is_local`: loopback hosts, unix sockets and an empty host are
/// local; anything else is refused (the connection is plaintext).
pub fn check_dsn_is_local(config: &postgres::Config) -> Result<(), PgError> {
    for h in config.get_hosts() {
        if let postgres::config::Host::Tcp(t) = h {
            let host = t.to_lowercase();
            if !["localhost", "127.0.0.1", "::1", ""].contains(&host.as_str()) {
                return Err(PgError::RemoteDsn { host });
            }
        }
    }
    Ok(())
}

/// pgvector `hnsw.max_scan_tuples` bound on an iterative scan (pgvector's own is 20000).
pub const DEFAULT_HNSW_MAX_SCAN_TUPLES: i64 = 10_000;

/// `hnsw_query_settings` plus `hnsw_max_scan_tuples` from skmemory (PRs 147 and
/// 148): the per-query `SET LOCAL` settings for a kNN of `limit` rows.
///
/// * `hnsw.ef_search`: `SKMEMORY_HNSW_EF_SEARCH` (default 200, garbage falls
///   back to the default, below 1 skips), never below `limit` (HNSW hands back
///   at most `ef_search` candidates before the WHERE clause runs) and capped at
///   pgvector's 1000.
/// * `hnsw.iterative_scan`: `SKMEMORY_HNSW_ITERATIVE_SCAN` (`off`,
///   `strict_order`, `relaxed_order`; default `relaxed_order`; anything else skips).
/// * `hnsw.max_scan_tuples`: `SKMEMORY_HNSW_MAX_SCAN_TUPLES` (default 10000,
///   garbage falls back, below 1 skips), set only when the iterative scan is
///   on (mode set and not `off`).
pub fn search_session_settings(
    get: impl Fn(&str) -> Option<String>,
    limit: i64,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let ef = match get("SKMEMORY_HNSW_EF_SEARCH") {
        None => DEFAULT_HNSW_EF_SEARCH,
        Some(raw) => raw.trim().parse::<i64>().unwrap_or(DEFAULT_HNSW_EF_SEARCH),
    };
    if ef >= 1 {
        out.push((
            "hnsw.ef_search".to_string(),
            ef.max(limit).min(1000).to_string(),
        ));
    }
    let mode = get("SKMEMORY_HNSW_ITERATIVE_SCAN")
        .unwrap_or_else(|| DEFAULT_HNSW_ITERATIVE_SCAN.to_string())
        .trim()
        .to_lowercase();
    let mode_ok = matches!(mode.as_str(), "off" | "strict_order" | "relaxed_order");
    if mode_ok {
        out.push(("hnsw.iterative_scan".to_string(), mode.clone()));
    }
    let max_scan = match get("SKMEMORY_HNSW_MAX_SCAN_TUPLES") {
        None => DEFAULT_HNSW_MAX_SCAN_TUPLES,
        Some(raw) => raw
            .trim()
            .parse::<i64>()
            .unwrap_or(DEFAULT_HNSW_MAX_SCAN_TUPLES),
    };
    if max_scan >= 1 && mode_ok && mode != "off" {
        out.push(("hnsw.max_scan_tuples".to_string(), max_scan.to_string()));
    }
    out
}

/// `_as_of_clause`: SQL predicate and the number of times the `as_of` bind is
/// used. `coalesce(valid_from, created_at) <= T AND (valid_to IS NULL OR
/// valid_to > T)`. Without the typed columns the same fields are read from
/// `memory_json`, cast only when they look like a timestamp. `p` is the
/// 1-based placeholder of the as_of text.
pub fn as_of_clause(v2_columns: bool, p: usize) -> String {
    let (lo, hi) = if v2_columns {
        ("valid_from".to_string(), "valid_to".to_string())
    } else {
        let iso = "'^[0-9]{4}-[0-9]{2}-[0-9]{2}'";
        (
            format!(
                "(CASE WHEN memory_json->>'valid_from' ~ {iso} THEN (memory_json->>'valid_from')::timestamptz END)"
            ),
            format!(
                "(CASE WHEN memory_json->>'valid_to' ~ {iso} THEN (memory_json->>'valid_to')::timestamptz END)"
            ),
        )
    };
    format!(
        "(COALESCE({lo}, created_at) <= ${p}::text::timestamptz AND ({hi} IS NULL OR {hi} > ${p}::text::timestamptz))"
    )
}

pub(crate) type Binds = Vec<Box<dyn ToSql + Sync>>;

/// Append a bind and return its 1-based placeholder number.
pub(crate) fn push_bind(binds: &mut Binds, b: Box<dyn ToSql + Sync>) -> usize {
    binds.push(b);
    binds.len()
}

/// The `WHERE` clauses every arm shares (agent, layer, tags AND, context_tag,
/// source), each value a bound parameter. `as_of` is the caller's business: it
/// differs per arm. An empty tag list, empty source and `None` are "no
/// filter"; an empty `context_tag` list matches nothing (fail closed).
pub(crate) fn filter_clauses(agent: &str, filters: &Filters, binds: &mut Binds) -> Vec<String> {
    let mut clauses = Vec::new();
    let p = push_bind(binds, Box::new(agent.to_string()));
    clauses.push(format!("agent=${p}"));
    if let Some(layer) = filters.layer {
        let p = push_bind(binds, Box::new(layer.as_str().to_string()));
        clauses.push(format!("layer=${p}"));
    }
    if !filters.tags.is_empty() {
        let p = push_bind(binds, Box::new(filters.tags.clone()));
        clauses.push(format!("tags @> ${p}"));
    }
    match &filters.context_tag {
        None => {}
        Some(ContextTag::One(t)) => {
            let p = push_bind(binds, Box::new(t.clone()));
            clauses.push(format!("memory_json->>'context_tag' = ${p}"));
        }
        Some(ContextTag::AnyOf(list)) => {
            let p = push_bind(binds, Box::new(list.clone()));
            clauses.push(format!("memory_json->>'context_tag' = ANY(${p})"));
        }
    }
    if let Some(src) = filters.source.as_deref().filter(|s| !s.is_empty()) {
        let p = push_bind(binds, Box::new(src.to_string()));
        clauses.push(format!("source=${p}"));
    }
    clauses
}

/// Whether the schema-v2 typed columns exist in the connection's schema.
pub(crate) fn probe_v2_columns(client: &mut Client) -> Result<bool, PgError> {
    let cols: Vec<String> = V2_COLUMNS.iter().map(|s| s.to_string()).collect();
    let row = client
        .query_one(
            "SELECT count(*) FROM information_schema.columns \
             WHERE table_schema = current_schema() AND table_name = 'memories' \
             AND column_name = ANY($1)",
            &[&cols],
        )
        .map_err(|e| PgError::Query(e.to_string()))?;
    let n: i64 = row.get(0);
    Ok(n == V2_COLUMNS.len() as i64)
}

/// Fail with [`PgError::EmptySearchPath`] when no schema on the effective
/// `search_path` exists. Run inside the query's transaction so it sees the
/// same path the query will.
pub(crate) fn check_search_path(tx: &mut postgres::Transaction<'_>) -> Result<(), PgError> {
    let row = tx
        .query_one("SELECT cardinality(current_schemas(false))::int8", &[])
        .map_err(|e| PgError::Query(e.to_string()))?;
    let n: i64 = row.get(0);
    if n == 0 {
        Err(PgError::EmptySearchPath)
    } else {
        Ok(())
    }
}

/// Parse a `memory_json` cell, `None` for a row that will not parse.
pub(crate) fn parse_memory_cell(json: Value) -> Option<Memory> {
    match json {
        Value::String(s) => Memory::from_stored_json(s.as_bytes()),
        other => Memory::from_stored(other),
    }
    .ok()
}

/// Connect from parsed settings with the loopback guard and a scrubbed error.
pub(crate) fn open_client(
    config: &postgres::Config,
    allow_remote: bool,
) -> Result<Client, PgError> {
    let password = config
        .get_password()
        .map(|p| String::from_utf8_lossy(p).to_string())
        .unwrap_or_default();
    if !allow_remote {
        check_dsn_is_local(config)?;
    }
    config
        .connect(NoTls)
        .map_err(|e| PgError::Connect(scrub(&e.to_string(), &[&password])))
}

/// `SKMEMORY_PG_ALLOW_REMOTE` as the vector arm reads it.
pub(crate) fn allow_remote_from_env() -> bool {
    matches!(
        std::env::var("SKMEMORY_PG_ALLOW_REMOTE")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

impl VectorArm {
    /// Connect with `SKMEMORY_PG_DSN`.
    pub fn from_env(agent: &str, embedder: Box<dyn Embedder + Send>) -> Result<VectorArm, PgError> {
        let dsn = std::env::var("SKMEMORY_PG_DSN").unwrap_or_default();
        if dsn.trim().is_empty() {
            return Err(PgError::NoDsn);
        }
        VectorArm::connect(&dsn, agent, embedder)
    }

    /// Connect (refusing a non-loopback host unless `SKMEMORY_PG_ALLOW_REMOTE`
    /// is set).
    pub fn connect(
        dsn: &str,
        agent: &str,
        embedder: Box<dyn Embedder + Send>,
    ) -> Result<VectorArm, PgError> {
        VectorArm::connect_with(dsn, agent, embedder, allow_remote_from_env())
    }

    /// Connect with an explicit remote-host decision.
    pub fn connect_with(
        dsn: &str,
        agent: &str,
        embedder: Box<dyn Embedder + Send>,
        allow_remote: bool,
    ) -> Result<VectorArm, PgError> {
        let config: postgres::Config = dsn
            .parse()
            .map_err(|_| PgError::Connect("invalid connection string".to_string()))?;
        VectorArm::connect_config(config, agent, embedder, allow_remote)
    }

    /// Connect from parsed settings (lets a caller add options such as
    /// `search_path`). The DSN text is not available here, so connect errors
    /// are scrubbed of the password by construction.
    pub fn connect_config(
        config: postgres::Config,
        agent: &str,
        embedder: Box<dyn Embedder + Send>,
        allow_remote: bool,
    ) -> Result<VectorArm, PgError> {
        let client = open_client(&config, allow_remote)?;
        Ok(VectorArm {
            client,
            agent: agent.to_string(),
            embedder,
            v2_columns: None,
            reconnect: Some(config),
            unparseable: 0,
        })
    }

    /// Wrap an already connected client (tests use this to pick a schema).
    /// The caller vouches for the host; such an arm cannot reconnect.
    pub fn from_client(
        client: Client,
        agent: &str,
        embedder: Box<dyn Embedder + Send>,
    ) -> Result<VectorArm, PgError> {
        Ok(VectorArm {
            client,
            agent: agent.to_string(),
            embedder,
            v2_columns: None,
            reconnect: None,
            unparseable: 0,
        })
    }

    /// Open a fresh connection from the stored settings.
    pub fn reconnect(&mut self) -> Result<(), PgError> {
        let config = self
            .reconnect
            .clone()
            .ok_or_else(|| PgError::Connect("no connection settings to reconnect with".into()))?;
        self.client = config
            .connect(NoTls)
            .map_err(|e| PgError::Connect(e.to_string()))?;
        self.v2_columns = None;
        Ok(())
    }

    /// Run `sql` in a transaction whose search settings are `SET LOCAL`
    /// (`set_config(..., true)`): they end with the transaction, so nothing
    /// leaks onto a connection the caller shares. An older pgvector rejects
    /// the hnsw settings: each is tried in a savepoint and skipped on error.
    fn run(
        &mut self,
        sql: &str,
        refs: &[&(dyn ToSql + Sync)],
        limit: i64,
    ) -> Result<Vec<postgres::Row>, postgres::Error> {
        let settings = search_session_settings(|k| std::env::var(k).ok(), limit);
        let mut tx = self.client.transaction()?;
        for (name, value) in settings {
            let mut sp = tx.transaction()?;
            if sp
                .query("SELECT set_config($1, $2, true)", &[&name, &value])
                .is_ok()
            {
                sp.commit()?;
            }
        }
        let rows = tx.query(sql, refs)?;
        tx.commit()?;
        Ok(rows)
    }

    fn has_v2_columns(&mut self) -> Result<bool, PgError> {
        if let Some(v) = self.v2_columns {
            return Ok(v);
        }
        let v = probe_v2_columns(&mut self.client)?;
        self.v2_columns = Some(v);
        Ok(v)
    }

    /// The underlying client (tests inspect session state through it).
    pub fn client_mut(&mut self) -> &mut Client {
        &mut self.client
    }

    /// The agent this arm is scoped to.
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Build the SQL and the bind list for a query vector and filters.
    fn plan(
        &mut self,
        qvec: Vec<f32>,
        filters: &Filters,
        limit: i64,
        as_of: Option<String>,
    ) -> Result<(String, Vec<Box<dyn ToSql + Sync>>), PgError> {
        let mut binds: Binds = vec![Box::new(Vector::from(qvec))];
        let mut clauses = filter_clauses(&self.agent, filters, &mut binds);
        clauses.insert(1, "embedding IS NOT NULL".to_string());
        if let Some(as_of) = as_of {
            let v2 = self.has_v2_columns()?;
            let p = push_bind(&mut binds, Box::new(as_of));
            clauses.push(as_of_clause(v2, p));
        }
        let pl = push_bind(&mut binds, Box::new(limit));
        let sql = format!(
            "WITH knn AS MATERIALIZED (SELECT memory_json, embedding <=> $1::vector AS _d \
             FROM memories WHERE {} ORDER BY embedding <=> $1::vector LIMIT ${pl}) \
             SELECT memory_json, _d FROM knn ORDER BY _d",
            clauses.join(" AND ")
        );
        Ok((sql, binds))
    }

    /// Dense cosine search. When the query cannot be embedded the arm
    /// abstains with an empty list rather than falling back to an unfiltered
    /// text search. A wrong-length vector is a typed error.
    pub fn search_semantic(
        &mut self,
        query: &str,
        limit: i64,
        filters: &Filters,
    ) -> Result<Vec<Hit>, PgError> {
        let qvec = self.embedder.embed(query, false)?;
        if qvec.is_empty() {
            return Ok(Vec::new());
        }
        self.search_vector(qvec, limit, filters)
    }

    /// Same, with the query vector already computed.
    pub fn search_vector(
        &mut self,
        qvec: Vec<f32>,
        limit: i64,
        filters: &Filters,
    ) -> Result<Vec<Hit>, PgError> {
        if qvec.len() != VECTOR_DIM {
            return Err(PgError::DimensionMismatch {
                expected: VECTOR_DIM,
                actual: qvec.len(),
            });
        }
        // Parse as_of first: a value Python rejects never reaches SQL.
        let as_of = match filters.as_of.as_deref() {
            Some(raw) => crate::validity::normalize_as_of(raw)?,
            None => None,
        };
        if self.client.is_closed() && self.reconnect.is_some() {
            self.reconnect()?;
        }
        let (sql, binds) = self.plan(qvec, filters, limit, as_of)?;
        let refs: Vec<&(dyn ToSql + Sync)> = binds.iter().map(|b| b.as_ref()).collect();
        let first = self.run(&sql, &refs, limit);
        let rows = match first {
            Err(e) if (e.is_closed() || self.client.is_closed()) && self.reconnect.is_some() => {
                self.reconnect()?;
                // the v2 probe is per connection: rebuild the plan's as_of clause
                self.run(&sql, &refs, limit)
            }
            other => other,
        }
        .map_err(|e| PgError::Query(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let json: Value = row.get(0);
            let dist: f64 = row.get(1);
            match parse_memory_cell(json) {
                Some(memory) => out.push(Hit {
                    memory,
                    distance: dist,
                }),
                None => self.unparseable += 1,
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_settings_defaults_and_overrides() {
        let none = |_: &str| None;
        let kv = |a: &str, b: &str| (a.to_string(), b.to_string());
        assert_eq!(
            search_session_settings(none, 5),
            vec![
                kv("hnsw.ef_search", "200"),
                kv("hnsw.iterative_scan", "relaxed_order"),
                kv("hnsw.max_scan_tuples", "10000"),
            ]
        );
        let env = |k: &str| match k {
            "SKMEMORY_HNSW_EF_SEARCH" => Some("0".to_string()),
            "SKMEMORY_HNSW_ITERATIVE_SCAN" => Some(" Strict_Order ".to_string()),
            "SKMEMORY_HNSW_MAX_SCAN_TUPLES" => Some("500".to_string()),
            _ => None,
        };
        assert_eq!(
            search_session_settings(env, 5),
            vec![
                kv("hnsw.iterative_scan", "strict_order"),
                kv("hnsw.max_scan_tuples", "500")
            ]
        );
        let junk = |k: &str| (k == "SKMEMORY_HNSW_EF_SEARCH").then(|| "abc".to_string());
        assert_eq!(search_session_settings(junk, 5)[0].1, "200");
        // ef_search is never below the limit and is capped at 1000
        assert_eq!(search_session_settings(none, 300)[0].1, "300");
        assert_eq!(search_session_settings(none, 5000)[0].1, "1000");
        let big = |k: &str| (k == "SKMEMORY_HNSW_EF_SEARCH").then(|| "5000".to_string());
        assert_eq!(search_session_settings(big, 5)[0].1, "1000");
        // max_scan_tuples is skipped when the iterative scan is off or unknown
        let off = |k: &str| (k == "SKMEMORY_HNSW_ITERATIVE_SCAN").then(|| "off".to_string());
        assert_eq!(search_session_settings(off, 5).len(), 2);
        let bad = |k: &str| (k == "SKMEMORY_HNSW_ITERATIVE_SCAN").then(|| "nope".to_string());
        assert_eq!(search_session_settings(bad, 5).len(), 1);
        let zero = |k: &str| (k == "SKMEMORY_HNSW_MAX_SCAN_TUPLES").then(|| "0".to_string());
        assert_eq!(search_session_settings(zero, 5).len(), 2);
    }

    #[test]
    fn loopback_check() {
        let ok = |d: &str| check_dsn_is_local(&d.parse().unwrap()).is_ok();
        assert!(ok(&format!("postgresql://u:p@{}:5432/db", "127.0.0.1")));
        assert!(ok("host=localhost user=u"));
        assert!(ok("postgresql://u@[::1]/db"));
        assert!(ok("host=/var/run/postgresql user=u"));
        assert!(!ok(&format!("postgresql://u:p@{}:5432/db", "192.0.2.158")));
        assert!(!ok("host=db.example.com user=u"));
        assert!(!ok("host=127.0.0.1,10.0.0.5 user=u"));
    }

    #[test]
    fn as_of_sql_shapes() {
        let v2 = as_of_clause(true, 4);
        assert!(v2.contains("COALESCE(valid_from, created_at) <= $4::text::timestamptz"));
        assert!(v2.contains("valid_to IS NULL OR valid_to > $4"));
        assert!(as_of_clause(false, 4).contains("memory_json->>'valid_from' ~"));
    }

    #[test]
    fn dsn_is_scrubbed() {
        let m = scrub(
            &format!("bad s3cret in x://u:{}@h/db", "s3cret"),
            &["s3cret", ""],
        );
        assert!(!m.contains("s3cret"));
        assert_eq!(scrub("abc", &[""]), "abc");
    }
}
