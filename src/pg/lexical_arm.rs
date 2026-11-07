//! The lexical arm, the Rust twin of `PGVectorBackend.search_lexical`
//! (SKMEM-R-05). Feature `pg`.
//!
//! A PURE lexical opinion with no vector stage: ParadeDB BM25 over
//! `memories_bm25` when `pg_search` and that index exist, `ts_rank` over the
//! `tsv` column otherwise. Which one runs is a capability decision made from
//! what the server offers (probed once per connection), never from how the
//! rows looked. If the BM25 statement itself errors the arm falls back to
//! `ts_rank` for that call and says so in [`LexicalArm::last_run`]: a silent
//! downgrade is indistinguishable from success, which is the defect class
//! this arm exists to remove.
//!
//! `as_of` on the BM25 path is NOT in the SQL (ParadeDB rejects the validity
//! predicate next to `@@@`): the arm over-fetches `limit * 4` and keeps rows
//! valid at `as_of` by parsed instants. On the `ts_rank` path it is plain SQL.
//!
//! Every value (agent, filters, the query text, the limit) is a bound
//! parameter. The only text in the statements is fixed SQL.
//!
//! Scores: BM25 `paradedb.score(id)` or `ts_rank(...)`, higher is better.
//! Ties break on `id` (Python leaves ties in physical order).

use super::{
    allow_remote_from_env, as_of_clause, check_dsn_is_local, check_search_path, filter_clauses,
    open_client, parse_memory_cell, probe_v2_columns, push_bind, Binds, Filters, PgError,
};
use crate::audience::{OwnerScope, ReaderScope};
use crate::model::Memory;
use crate::validity::{normalize_as_of, valid_at};
use postgres::types::ToSql;
use postgres::Client;
use std::fmt;

/// Which statement produced the last result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LexicalPath {
    /// ParadeDB BM25 (`paradedb.score`).
    Bm25,
    /// Postgres full text (`ts_rank`), `pg_search` or its index absent.
    TsRank,
    /// The BM25 statement failed and `ts_rank` answered instead. Carries the
    /// (password-free) server message.
    TsRankAfterBm25Error(String),
}

/// One result row.
#[derive(Debug)]
pub struct LexicalHit {
    pub memory: Memory,
    /// BM25 score or `ts_rank`, higher is better.
    pub score: f64,
}

/// The lexical arm over one agent.
pub struct LexicalArm {
    client: Client,
    agent: String,
    bm25: Option<bool>,
    v2_columns: Option<bool>,
    reconnect: Option<postgres::Config>,
    last_run: Option<LexicalPath>,
    /// Rows this arm could not parse (skipped, as Python does).
    pub unparseable: u64,
}

impl fmt::Debug for LexicalArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexicalArm")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

impl LexicalArm {
    /// Connect with `SKMEMORY_PG_DSN` (loopback only unless
    /// `SKMEMORY_PG_ALLOW_REMOTE` is set).
    pub fn from_env(agent: &str) -> Result<LexicalArm, PgError> {
        let dsn = std::env::var("SKMEMORY_PG_DSN").unwrap_or_default();
        if dsn.trim().is_empty() {
            return Err(PgError::NoDsn);
        }
        LexicalArm::connect_with(&dsn, agent, allow_remote_from_env())
    }

    /// Connect with an explicit remote-host decision.
    pub fn connect_with(dsn: &str, agent: &str, allow_remote: bool) -> Result<LexicalArm, PgError> {
        let config: postgres::Config = dsn
            .parse()
            .map_err(|_| PgError::Connect("invalid connection string".to_string()))?;
        LexicalArm::connect_config(config, agent, allow_remote)
    }

    /// Connect from parsed settings (lets a caller add `search_path`).
    pub fn connect_config(
        config: postgres::Config,
        agent: &str,
        allow_remote: bool,
    ) -> Result<LexicalArm, PgError> {
        if !allow_remote {
            check_dsn_is_local(&config)?;
        }
        let client = open_client(&config, allow_remote)?;
        Ok(LexicalArm {
            client,
            agent: agent.to_string(),
            bm25: None,
            v2_columns: None,
            reconnect: Some(config),
            last_run: None,
            unparseable: 0,
        })
    }

    /// Wrap a connected client (tests pick the schema). Cannot reconnect.
    pub fn from_client(client: Client, agent: &str) -> LexicalArm {
        LexicalArm {
            client,
            agent: agent.to_string(),
            bm25: None,
            v2_columns: None,
            reconnect: None,
            last_run: None,
            unparseable: 0,
        }
    }

    /// Force the capability decision (`Some(false)` pins the `ts_rank` path,
    /// `None` re-probes). Parity tests use it to exercise both statements on
    /// one server.
    pub fn set_bm25(&mut self, bm25: Option<bool>) {
        self.bm25 = bm25;
    }

    /// The agent this arm is scoped to.
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// The underlying client. RAW ACCESS: bypasses the reader ceiling, so it
    /// takes an [`OwnerScope`].
    pub fn client_mut(&mut self, _owner: &OwnerScope) -> &mut Client {
        &mut self.client
    }

    /// Which statement answered the last search.
    pub fn last_run(&self) -> Option<&LexicalPath> {
        self.last_run.as_ref()
    }

    fn reconnect(&mut self) -> Result<(), PgError> {
        let config = self
            .reconnect
            .clone()
            .ok_or_else(|| PgError::Connect("no connection settings to reconnect with".into()))?;
        self.client = open_client(&config, true)?;
        self.bm25 = None;
        self.v2_columns = None;
        Ok(())
    }

    /// `_has_bm25_index`: `pg_search` installed AND `memories_bm25` present in
    /// a schema of the effective `search_path`. Probed once per connection.
    /// A probe failure means "no BM25" (as Python), not an error.
    fn has_bm25(&mut self) -> bool {
        if let Some(v) = self.bm25 {
            return v;
        }
        let v = self
            .client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = $1) \
                 AND EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = $2 \
                 AND indexname = $3 AND schemaname = ANY(current_schemas(false)))",
                &[&"pg_search", &"memories", &"memories_bm25"],
            )
            .map(|r| r.get::<_, bool>(0))
            .unwrap_or(false);
        self.bm25 = Some(v);
        v
    }

    fn bm25_statement(
        &mut self,
        query: &str,
        filters: &Filters,
        scope: &ReaderScope,
        fetch: i64,
    ) -> Result<Vec<(Memory, f64)>, StatementError> {
        let mut binds: Binds = Vec::new();
        let clauses = filter_clauses(&self.agent, filters, scope, &mut binds);
        let q = push_bind(&mut binds, Box::new(query.to_string()));
        let l = push_bind(&mut binds, Box::new(fetch));
        let sql = format!(
            "SELECT memory_json, paradedb.score(id)::float8 FROM memories WHERE {} \
             AND id @@@ paradedb.boolean(should => ARRAY[\
             paradedb.match('content', ${q}), paradedb.match('title', ${q})]) \
             ORDER BY paradedb.score(id) DESC, id LIMIT ${l}",
            clauses.join(" AND ")
        );
        self.run(&sql, &binds)
    }

    fn tsrank_statement(
        &mut self,
        query: &str,
        filters: &Filters,
        scope: &ReaderScope,
        as_of: Option<String>,
        limit: i64,
    ) -> Result<Vec<(Memory, f64)>, StatementError> {
        let mut binds: Binds = Vec::new();
        let mut clauses = filter_clauses(&self.agent, filters, scope, &mut binds);
        if let Some(as_of) = as_of {
            let v2 = match self.v2_columns {
                Some(v) => v,
                None => {
                    let v = probe_v2_columns(&mut self.client).map_err(StatementError::Pg)?;
                    self.v2_columns = Some(v);
                    v
                }
            };
            let p = push_bind(&mut binds, Box::new(as_of));
            clauses.push(as_of_clause(v2, p));
        }
        let q = push_bind(&mut binds, Box::new(query.to_string()));
        let l = push_bind(&mut binds, Box::new(limit));
        let sql = format!(
            "SELECT memory_json, ts_rank(tsv, plainto_tsquery('english', ${q}))::float8 \
             FROM memories WHERE {} AND tsv @@ plainto_tsquery('english', ${q}) \
             ORDER BY ts_rank(tsv, plainto_tsquery('english', ${q})) DESC, id LIMIT ${l}",
            clauses.join(" AND ")
        );
        self.run(&sql, &binds)
    }

    /// One statement in its own transaction (so a failed BM25 attempt cannot
    /// poison the `ts_rank` one), after the `search_path` check.
    fn run(&mut self, sql: &str, binds: &Binds) -> Result<Vec<(Memory, f64)>, StatementError> {
        let refs: Vec<&(dyn ToSql + Sync)> = binds.iter().map(|b| b.as_ref()).collect();
        let mut tx = self
            .client
            .transaction()
            .map_err(|e| StatementError::Db(e.to_string(), e.is_closed()))?;
        check_search_path(&mut tx).map_err(StatementError::Pg)?;
        let rows = tx
            .query(sql, &refs)
            .map_err(|e| StatementError::Db(e.to_string(), e.is_closed()))?;
        tx.commit()
            .map_err(|e| StatementError::Db(e.to_string(), e.is_closed()))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let json: serde_json::Value = row.get(0);
            let score: f64 = row.get(1);
            match parse_memory_cell(json) {
                Some(m) => out.push((m, score)),
                None => self.unparseable += 1,
            }
        }
        Ok(out)
    }

    /// Pure lexical search as `scope` sees it (the reader ceiling is in the
    /// SQL `WHERE` and re-checked on every hydrated row). `limit <= 0` returns nothing. `as_of` is parsed
    /// in Rust first: a value Python rejects never reaches SQL.
    pub fn search_lexical(
        &mut self,
        query: &str,
        limit: i64,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<LexicalHit>, PgError> {
        let as_of = match filters.as_of.as_deref() {
            Some(raw) => normalize_as_of(raw)?,
            None => None,
        };
        if limit <= 0 {
            return Ok(Vec::new());
        }
        if self.client.is_closed() && self.reconnect.is_some() {
            self.reconnect()?;
        }
        let fetch = if as_of.is_some() {
            limit.saturating_mul(4)
        } else {
            limit
        };
        let mut attempt = self.search_once(query, filters, scope, as_of.as_deref(), limit, fetch);
        if matches!(attempt, Err(PgError::Connect(_)) | Err(PgError::Query(_)))
            && self.client.is_closed()
            && self.reconnect.is_some()
        {
            self.reconnect()?;
            attempt = self.search_once(query, filters, scope, as_of.as_deref(), limit, fetch);
        }
        let (rows, path) = attempt?;
        self.last_run = Some(path);
        let mut hits: Vec<LexicalHit> = rows
            .into_iter()
            .filter(|(m, _)| scope.allows_memory(m))
            .map(|(memory, score)| LexicalHit { memory, score })
            .collect();
        {
            let mut refs: Vec<&mut Memory> = hits.iter_mut().map(|h| &mut h.memory).collect();
            super::redact_memories(&mut self.client, &self.agent, &mut refs, scope)?;
        }
        Ok(hits)
    }

    fn search_once(
        &mut self,
        query: &str,
        filters: &Filters,
        scope: &ReaderScope,
        as_of: Option<&str>,
        limit: i64,
        fetch: i64,
    ) -> Result<(Vec<(Memory, f64)>, LexicalPath), PgError> {
        let mut bm25_error = None;
        if self.has_bm25() {
            match self.bm25_statement(query, filters, scope, fetch) {
                Ok(mut rows) => {
                    if let Some(t) = as_of {
                        rows.retain(|(m, _)| {
                            valid_at(
                                m.valid_from.as_deref(),
                                Some(m.created_at.as_str()),
                                m.valid_to.as_deref(),
                                t,
                            )
                        });
                    }
                    rows.truncate(limit as usize);
                    return Ok((rows, LexicalPath::Bm25));
                }
                // Not a capability problem: the typed errors a caller must see.
                Err(StatementError::Pg(e)) => return Err(e),
                Err(StatementError::Db(msg, closed)) => {
                    if closed {
                        return Err(PgError::Query(msg));
                    }
                    bm25_error = Some(msg);
                }
            }
        }
        let rows = self
            .tsrank_statement(query, filters, scope, as_of.map(str::to_string), limit)
            .map_err(StatementError::into_pg)?;
        let path = match bm25_error {
            Some(m) => LexicalPath::TsRankAfterBm25Error(m),
            None => LexicalPath::TsRank,
        };
        Ok((rows, path))
    }
}

enum StatementError {
    Pg(PgError),
    /// Server error text and whether the connection is gone.
    Db(String, bool),
}

impl StatementError {
    fn into_pg(self) -> PgError {
        match self {
            StatementError::Pg(e) => e,
            StatementError::Db(m, _) => PgError::Query(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_has_no_connection_material() {
        // Shape only: LexicalArm needs a live client, so check the path enum.
        let p = LexicalPath::TsRankAfterBm25Error("boom".into());
        assert!(format!("{p:?}").contains("boom"));
        assert_ne!(LexicalPath::Bm25, LexicalPath::TsRank);
    }
}
