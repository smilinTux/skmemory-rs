//! The graph arm over Apache AGE, the Rust twin of the search and temporal
//! reads of `AGEGraphBackend` (SKMEM-R-05). Feature `pg`.
//!
//! * **Off by default.** The arm always exists and always answers
//!   [`GraphArm::state`], but [`GraphArm::search`] only runs when
//!   `SKMEMORY_GRAPH_ARM` is `on`/`1`/`true`/`yes`; otherwise it reports
//!   `disabled by config (SKMEMORY_GRAPH_ARM is off, the default)` and opens
//!   no connection and runs no statement. Lineage, neighbour and validity
//!   reads are separate methods and are never gated by that flag, as in
//!   Python ("this switches off ONLY the search arm").
//! * **Every value is bound.** AGE takes its parameters as one agtype map
//!   (`cypher(name, $$ ... $$, $1)`); the Cypher text is a constant, with the
//!   relationship type taken only from the fixed [`MEMORY_LINK_EDGES`] list.
//!   The graph name is the single inline piece (AGE requires a name constant),
//!   so it is validated as a plain identifier at construction and an invalid
//!   name disables the arm. Even `statement_timeout` and `search_path` travel
//!   through `set_config(name, $1, true)`.
//! * **The empty `search_path` trap.** AGE needs `ag_catalog` on the path
//!   (without it `create_graph` fails with `operator class "graphid_ops" does
//!   not exist`, which never mentions the path). Each query therefore runs in
//!   a transaction that sets `search_path = ag_catalog, "$user", public` for
//!   that transaction only, so an empty or foreign path on the connection
//!   neither breaks the arm nor leaks the change onto the caller's session.
//! * **Time is compared as instants**, never as text: stored timestamps are
//!   strings in mixed forms (`Z`, `+00:00`, `+02:00`), so a lexical compare is
//!   wrong at the boundary in both directions. The Cypher carries only a
//!   coarse date prefilter (a superset), and the exact rule runs in Rust on
//!   parsed values ([`crate::validity`]).

use super::{
    allow_remote_from_env, check_dsn_is_local, open_client, scrub, ContextTag, Filters, PgError,
};
use crate::audience::{OwnerScope, ReaderScope};
use crate::model::py_strip;
use crate::util::{iso_from_unix, now_iso};
use crate::validity::{normalize_as_of, parse_ts, valid_at};
use postgres::error::SqlState;
use postgres::types::private::BytesMut;
use postgres::types::{FromSql, IsNull, ToSql, Type};
use postgres::Client;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

/// Env switch for the graph search arm. OFF unless set to a true word.
pub const GRAPH_ARM_ENV: &str = "SKMEMORY_GRAPH_ARM";
/// Env cap on a graph SEARCH statement, milliseconds (default 2500, 0 or
/// negative disables). Mutation and reconcile reads are never capped.
pub const GRAPH_SEARCH_TIMEOUT_ENV: &str = "SKMEMORY_GRAPH_SEARCH_TIMEOUT_MS";
/// Default for [`GRAPH_SEARCH_TIMEOUT_ENV`].
pub const DEFAULT_SEARCH_TIMEOUT_MS: i64 = 2500;

/// Memory-to-Memory edge types, in Python's order. Deliberately not the hub
/// edges (`TAGGED_WITH`, `FROM_SOURCE`, `MENTIONS`): walking those explodes
/// through shared tags and entities.
pub const MEMORY_LINK_EDGES: [&str; 5] = [
    "RELATED_TO",
    "SUPERSEDES",
    "PROMOTED_FROM",
    "CHUNK_OF",
    "EXTENDS",
];
/// Edges walked outward for ancestry.
pub const LINEAGE_WALK_EDGES: [&str; 3] = ["PROMOTED_FROM", "CHUNK_OF", "SUPERSEDES"];

/// Query words too common to be worth an entity lookup (Python's list).
const TOKEN_STOPWORDS: [&str; 56] = [
    "what", "when", "where", "which", "about", "from", "with", "that", "this", "there", "their",
    "they", "then", "than", "have", "has", "had", "did", "does", "doing", "said", "says", "the",
    "and", "for", "was", "were", "been", "being", "into", "over", "under", "after", "before",
    "would", "could", "should", "will", "shall", "much", "many", "some", "any", "all", "only",
    "just", "also", "more", "most", "like", "want", "need", "know", "think", "make", "made",
];

const AG_SEARCH_PATH: &str = "ag_catalog, \"$user\", public";

/// Whether the graph search arm is on, from an env reader.
pub fn graph_arm_enabled(get: impl Fn(&str) -> Option<String>) -> bool {
    let mode = get(GRAPH_ARM_ENV)
        .unwrap_or_else(|| "off".to_string())
        .trim()
        .to_lowercase();
    matches!(mode.as_str(), "on" | "1" | "true" | "yes")
}

/// `_search_timeout_ms` from an env reader.
pub fn search_timeout_ms(get: impl Fn(&str) -> Option<String>) -> i64 {
    match get(GRAPH_SEARCH_TIMEOUT_ENV) {
        Some(raw) if !raw.trim().is_empty() => raw
            .trim()
            .parse::<i64>()
            .unwrap_or(DEFAULT_SEARCH_TIMEOUT_MS),
        _ => DEFAULT_SEARCH_TIMEOUT_MS,
    }
}

/// `^[A-Za-z_][A-Za-z0-9_]*$`.
pub fn valid_graph_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// What the arm reports about itself (`search_arm_states` in Python).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmState {
    pub enabled: bool,
    /// `enabled`, `disabled` or `unavailable`.
    pub state: &'static str,
    pub reason: Option<String>,
}

/// One entity-search row.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityHit {
    pub id: Option<String>,
    pub title: Option<String>,
    pub layer: Option<String>,
    pub entity: Option<String>,
}

/// A Memory neighbour, ancestor or valid-memory stub.
#[derive(Clone, Debug, PartialEq)]
pub struct Stub {
    pub id: Option<String>,
    pub title: Option<String>,
    pub layer: Option<String>,
    /// Hops (`get_related`, `neighbors`) or chain depth (lineage).
    pub distance: Option<i64>,
    /// `created_at` (only on `currently_valid_memories`).
    pub created_at: Option<String>,
}

/// Result of [`GraphArm::search`].
#[derive(Debug)]
pub struct ArmOutcome {
    pub state: ArmState,
    /// Empty when the arm is not enabled.
    pub hits: Vec<EntityHit>,
}

/// `agtype` as a query parameter: a one-byte version (1) and the JSON text.
#[derive(Debug)]
struct AgParams(String);

impl ToSql for AgParams {
    fn to_sql(
        &self,
        _: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
        out.extend_from_slice(&[1]);
        out.extend_from_slice(self.0.as_bytes());
        Ok(IsNull::No)
    }
    fn accepts(ty: &Type) -> bool {
        ty.name() == "agtype"
    }
    postgres::types::to_sql_checked!();
}

/// `agtype` result cell: the same wire shape, parsed to JSON. The
/// `::vertex`/`::edge`/`::path` suffix AGE appends is stripped.
struct Agtype(Value);

impl<'a> FromSql<'a> for Agtype {
    fn from_sql(_: &Type, raw: &'a [u8]) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let body = match raw.split_first() {
            Some((1, rest)) => rest,
            _ => raw,
        };
        let text = std::str::from_utf8(body)?.trim();
        let text = ["::vertex", "::edge", "::path"]
            .iter()
            .find_map(|s| text.strip_suffix(s))
            .unwrap_or(text)
            .trim();
        if text.is_empty() {
            return Ok(Agtype(Value::Null));
        }
        // Malformed agtype is "no value", never an error (Python: `_agtype`).
        Ok(Agtype(serde_json::from_str(text).unwrap_or(Value::Null)))
    }
    fn accepts(ty: &Type) -> bool {
        ty.name() == "agtype"
    }
}

/// The graph arm over one agent's graph.
pub struct GraphArm {
    client: Option<Client>,
    config: Option<postgres::Config>,
    allow_remote: bool,
    agent: String,
    graph: Option<String>,
    enabled: bool,
    loaded: bool,
    search_timeout_ms: i64,
}

impl fmt::Debug for GraphArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphArm")
            .field("agent", &self.agent)
            .field("graph", &self.graph)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

fn str_of(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string)
}

/// The UTC date of an instant (micros since the epoch) plus `days`, as
/// `YYYY-MM-DD`.
fn date_plus(micros: i128, days: i64) -> String {
    let secs = micros.div_euclid(1_000_000) as i64;
    let day_start = secs.div_euclid(86_400) * 86_400 + days * 86_400;
    iso_from_unix(day_start, 0)[..10].to_string()
}

/// `re.escape` as Python 3.7+ does it: a backslash before each of
/// `()[]{}?*+-|^$\.&~# ` and the ASCII whitespace controls. Postgres ARE
/// treats a backslash before any of those as the literal character.
fn py_re_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// `[A-Za-z][A-Za-z0-9_.-]{3,}` matches, left to right, non-overlapping.
fn find_tokens(q: &str) -> Vec<String> {
    let c: Vec<char> = q.chars().collect();
    let class = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-');
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_alphabetic() {
            let mut j = i + 1;
            while j < c.len() && class(c[j]) {
                j += 1;
            }
            if j - i > 3 {
                out.push(c[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn is_timeout(e: &PgError) -> bool {
    matches!(e, PgError::Timeout(_))
}

impl GraphArm {
    /// Build an arm that connects lazily on first use. `graph` defaults to
    /// `<agent>_knowledge`; an invalid name leaves the arm unavailable
    /// (every query returns [`PgError::BadGraphName`]). `enabled` is the
    /// search-arm switch (see [`graph_arm_enabled`]).
    pub fn new(
        config: postgres::Config,
        agent: &str,
        graph: Option<&str>,
        enabled: bool,
        allow_remote: bool,
    ) -> Result<GraphArm, PgError> {
        if !allow_remote {
            check_dsn_is_local(&config)?;
        }
        let mut arm = GraphArm::bare(agent, graph, enabled);
        arm.config = Some(config);
        arm.allow_remote = allow_remote;
        Ok(arm)
    }

    /// Same, from `SKMEMORY_PG_DSN` and `SKMEMORY_GRAPH_ARM`.
    pub fn from_env(agent: &str, graph: Option<&str>) -> Result<GraphArm, PgError> {
        let dsn = std::env::var("SKMEMORY_PG_DSN").unwrap_or_default();
        if dsn.trim().is_empty() {
            return Err(PgError::NoDsn);
        }
        let config: postgres::Config = dsn
            .parse()
            .map_err(|_| PgError::Connect("invalid connection string".to_string()))?;
        let enabled = graph_arm_enabled(|k| std::env::var(k).ok());
        GraphArm::new(config, agent, graph, enabled, allow_remote_from_env())
    }

    /// Wrap a connected client (tests). It cannot reconnect.
    pub fn from_client(
        client: Client,
        agent: &str,
        graph: Option<&str>,
        enabled: bool,
    ) -> GraphArm {
        let mut arm = GraphArm::bare(agent, graph, enabled);
        arm.client = Some(client);
        arm
    }

    fn bare(agent: &str, graph: Option<&str>, enabled: bool) -> GraphArm {
        let name = graph
            .map(str::to_string)
            .unwrap_or_else(|| format!("{agent}_knowledge"));
        GraphArm {
            client: None,
            config: None,
            allow_remote: false,
            agent: agent.to_string(),
            graph: valid_graph_name(&name).then_some(name),
            enabled,
            loaded: false,
            search_timeout_ms: search_timeout_ms(|k| std::env::var(k).ok()),
        }
    }

    /// The connected client, if any (tests inspect session state through it).
    /// RAW ACCESS: bypasses the reader ceiling, as [`GraphArm::cypher`] does,
    /// so it takes an [`OwnerScope`].
    pub fn client_mut(&mut self, _owner: &OwnerScope) -> Option<&mut Client> {
        self.client.as_mut()
    }

    /// Override the search statement cap (0 or negative disables it).
    pub fn set_search_timeout_ms(&mut self, ms: i64) {
        self.search_timeout_ms = ms;
    }

    /// The agent this arm is scoped to.
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// The validated graph name, `None` when the requested one was refused.
    pub fn graph(&self) -> Option<&str> {
        self.graph.as_deref()
    }

    /// `search_arm_states()["graph"]`.
    pub fn state(&self) -> ArmState {
        if self.graph.is_none() {
            ArmState {
                enabled: false,
                state: "unavailable",
                reason: Some("invalid graph name".to_string()),
            }
        } else if self.enabled {
            ArmState {
                enabled: true,
                state: "enabled",
                reason: None,
            }
        } else {
            ArmState {
                enabled: false,
                state: "disabled",
                reason: Some(format!(
                    "disabled by config ({GRAPH_ARM_ENV} is off, the default)"
                )),
            }
        }
    }

    fn client(&mut self) -> Result<&mut Client, PgError> {
        let dead = self.client.as_ref().is_none_or(|c| c.is_closed());
        if dead {
            let config = self
                .config
                .clone()
                .ok_or_else(|| PgError::Connect("no connection settings".to_string()))?;
            self.client = Some(open_client(&config, self.allow_remote)?);
            self.loaded = false;
        }
        Ok(self.client.as_mut().expect("client just set"))
    }

    /// Run one Cypher statement and return every row as `ncols` JSON values.
    ///
    /// `query` must be a constant (it is placed between `$$` quotes); every
    /// caller value goes in `params`. `timeout_ms` caps this statement only
    /// (`SET LOCAL` semantics). Public so a seeding harness can write the
    /// graph with the same bound-parameter path. RAW ACCESS: this bypasses the
    /// reader ceiling and returns whatever the Cypher selects. It is a
    /// seeding and diagnostics handle for the owner, never a search path.
    pub fn cypher(
        &mut self,
        query: &str,
        params: &Value,
        ncols: usize,
        timeout_ms: Option<i64>,
        _owner: &OwnerScope,
    ) -> Result<Vec<Vec<Value>>, PgError> {
        self.cypher_raw(query, params, ncols, timeout_ms)
    }

    /// The crate's own constant-query path (see [`GraphArm::cypher`]).
    pub(crate) fn cypher_raw(
        &mut self,
        query: &str,
        params: &Value,
        ncols: usize,
        timeout_ms: Option<i64>,
    ) -> Result<Vec<Vec<Value>>, PgError> {
        let graph = self
            .graph
            .clone()
            .ok_or_else(|| PgError::BadGraphName(format!("{}_knowledge", self.agent)))?;
        if query.contains("$$") {
            return Err(PgError::Query("cypher text may not contain $$".to_string()));
        }
        let cols: Vec<String> = (0..ncols.max(1)).map(|i| format!("c{i} agtype")).collect();
        let sql = format!(
            "SELECT * FROM ag_catalog.cypher('{graph}', $$ {query} $$, $1) AS ({})",
            cols.join(", ")
        );
        let payload = AgParams(params.to_string());
        let loaded = self.loaded;
        let client = self.client()?;
        if !loaded {
            client
                .batch_execute("LOAD 'age'")
                .map_err(|e| PgError::Query(format!("LOAD 'age' failed: {e}")))?;
        }
        let map_err = |e: postgres::Error| {
            let msg = scrub(&e.to_string(), &[]);
            if e.code() == Some(&SqlState::QUERY_CANCELED) {
                PgError::Timeout(msg)
            } else {
                PgError::Query(msg)
            }
        };
        let mut tx = client.transaction().map_err(map_err)?;
        tx.query_one(
            "SELECT set_config('search_path', $1, true)",
            &[&AG_SEARCH_PATH],
        )
        .map_err(map_err)?;
        if let Some(ms) = timeout_ms.filter(|m| *m > 0) {
            tx.query_one(
                "SELECT set_config('statement_timeout', $1, true)",
                &[&ms.to_string()],
            )
            .map_err(map_err)?;
        }
        let rows = tx.query(sql.as_str(), &[&payload]).map_err(map_err)?;
        tx.commit().map_err(map_err)?;
        self.loaded = true;
        Ok(rows
            .iter()
            .map(|r| {
                (0..ncols)
                    .map(|i| r.get::<_, Option<Agtype>>(i).map_or(Value::Null, |a| a.0))
                    .collect()
            })
            .collect())
    }

    /// `_memory_filter`: a ` WHERE ...` fragment over `m` and its parameters.
    /// Only generated names (`$layer`, `$tag0`) enter the text. `$tagN IN
    /// m.tags` is used because AGE does not reliably support `ALL()`. A node
    /// with no `context_tag` never matches a context filter (fails closed).
    fn memory_filter(filters: &Filters) -> Result<(String, Map<String, Value>), PgError> {
        let mut parts: Vec<String> = Vec::new();
        let mut params = Map::new();
        if let Some(layer) = filters.layer {
            parts.push("m.layer = $layer".to_string());
            params.insert("layer".into(), json!(layer.as_str()));
        }
        for (i, tag) in filters.tags.iter().enumerate() {
            parts.push(format!("$tag{i} IN m.tags"));
            params.insert(format!("tag{i}"), json!(tag));
        }
        match &filters.context_tag {
            None => {}
            Some(ContextTag::One(t)) => {
                parts.push("m.context_tag = $context_tag".to_string());
                params.insert("context_tag".into(), json!(t));
            }
            Some(ContextTag::AnyOf(list)) => {
                parts.push("m.context_tag IN $context_tags".to_string());
                params.insert("context_tags".into(), json!(list));
            }
        }
        if let Some(src) = filters.source.as_deref().filter(|s| !s.is_empty()) {
            parts.push("m.source = $source".to_string());
            params.insert("source".into(), json!(src));
        }
        if let Some(as_of) = filters.as_of.as_deref() {
            if let Some(t) = normalize_as_of(as_of)?.and_then(|n| parse_ts(&n)) {
                // COARSE pre-filter only: stored values come in mixed offsets,
                // so compare against date bounds wide enough for any offset (a
                // false keep is fine, a false drop is not); the exact rule runs
                // in Rust afterwards.
                parts.push(
                    "coalesce(m.valid_from, m.created_at) < $as_of_hi \
                     AND (m.valid_to IS NULL OR m.valid_to >= $as_of_lo)"
                        .to_string(),
                );
                params.insert("as_of_hi".into(), json!(date_plus(t, 2)));
                params.insert("as_of_lo".into(), json!(date_plus(t, -1)));
            }
        }
        let where_ = if parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", parts.join(" AND "))
        };
        Ok((where_, params))
    }

    // ------------------------------------------------------------------
    // reader-scoped entry points (SKMEM-R-06C)
    //
    // Graph nodes carry no trustworthy audience, so for a restricted reader
    // visibility is decided against `memories` in the same database (the
    // authoritative `memory_json`): an id with no row there is hidden (fail
    // closed). A hidden id reads exactly like a missing one in every result.
    // ------------------------------------------------------------------

    /// Over-fetch factor for a restricted reader, so hiding rows does not
    /// silently shorten a page.
    const RESTRICTED_FETCH: usize = 4;

    /// Which of `ids` the reader may see. The owner sees all of them.
    fn visible_ids(
        &mut self,
        ids: &[String],
        scope: &ReaderScope,
    ) -> Result<HashSet<String>, PgError> {
        if scope.unrestricted() || ids.is_empty() {
            return Ok(ids.iter().cloned().collect());
        }
        let agent = self.agent.clone();
        super::visible_ids(self.client()?, &agent, ids, scope)
    }

    fn can_see(&mut self, id: &str, scope: &ReaderScope) -> Result<bool, PgError> {
        Ok(self.visible_ids(&[id.to_string()], scope)?.contains(id))
    }

    /// Entity search as `scope` sees it.
    pub fn search_by_entity(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<EntityHit>, PgError> {
        if scope.unrestricted() {
            return self.search_by_entity_raw(query, limit, filters);
        }
        let hits = self.search_by_entity_raw(
            query,
            limit.saturating_mul(Self::RESTRICTED_FETCH),
            filters,
        )?;
        let ids: Vec<String> = hits.iter().filter_map(|h| h.id.clone()).collect();
        let vis = self.visible_ids(&ids, scope)?;
        let mut out: Vec<EntityHit> = hits
            .into_iter()
            .filter(|h| h.id.as_ref().is_some_and(|i| vis.contains(i)))
            .collect();
        out.truncate(limit);
        Ok(out)
    }

    /// Memories valid at `as_of`, as `scope` sees them.
    pub fn currently_valid_memories(
        &mut self,
        as_of: Option<&str>,
        limit: usize,
        scope: &ReaderScope,
    ) -> Result<Vec<Stub>, PgError> {
        if scope.unrestricted() {
            return self.currently_valid_memories_raw(as_of, limit);
        }
        let rows =
            self.currently_valid_memories_raw(as_of, limit.saturating_mul(Self::RESTRICTED_FETCH))?;
        let ids: Vec<String> = rows.iter().filter_map(|s| s.id.clone()).collect();
        let vis = self.visible_ids(&ids, scope)?;
        let mut out: Vec<Stub> = rows
            .into_iter()
            .filter(|s| s.id.as_ref().is_some_and(|i| vis.contains(i)))
            .collect();
        out.truncate(limit);
        Ok(out)
    }

    /// Whether `memory_id` is valid at `as_of`. A hidden id is `false`, like a
    /// missing one.
    pub fn is_currently_valid(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
        scope: &ReaderScope,
    ) -> Result<bool, PgError> {
        if !scope.unrestricted() && !self.can_see(memory_id, scope)? {
            // Still validate `as_of` so a bad value errors the same either way.
            Self::resolve_ts(as_of)?;
            return Ok(false);
        }
        self.is_currently_valid_raw(memory_id, as_of)
    }

    /// Direct neighbourhood as `scope` sees it: a hidden start is empty, hidden
    /// neighbours are dropped before the cap.
    pub fn neighbors(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
        scope: &ReaderScope,
    ) -> Result<Vec<Stub>, PgError> {
        if scope.unrestricted() {
            return self.neighbors_raw(memory_id, as_of, 50);
        }
        if !self.can_see(memory_id, scope)? {
            Self::resolve_ts(as_of)?;
            return Ok(Vec::new());
        }
        let rows = self.neighbors_raw(memory_id, as_of, usize::MAX)?;
        let ids: Vec<String> = rows.iter().filter_map(|s| s.id.clone()).collect();
        let vis = self.visible_ids(&ids, scope)?;
        let mut out: Vec<Stub> = rows
            .into_iter()
            .filter(|s| s.id.as_ref().is_some_and(|i| vis.contains(i)))
            .collect();
        out.truncate(50);
        Ok(out)
    }

    /// The `SUPERSEDES` lineage as `scope` sees it: a hidden start is empty and
    /// a hidden ancestor is cut (neither returned nor walked through).
    pub fn supersedes_chain(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
        scope: &ReaderScope,
    ) -> Result<Vec<Stub>, PgError> {
        if scope.unrestricted() {
            return self.supersedes_chain_raw(memory_id, as_of, &mut |_, _| Ok(true));
        }
        if !self.can_see(memory_id, scope)? {
            Self::resolve_ts(as_of)?;
            return Ok(Vec::new());
        }
        self.supersedes_chain_raw(memory_id, as_of, &mut |arm, id| arm.can_see(id, scope))
    }

    /// Relatives up to `depth` hops as `scope` sees them: walked one hop at a
    /// time and never through a hidden node (Python `graph_related`).
    pub fn get_related(
        &mut self,
        memory_id: &str,
        depth: i64,
        scope: &ReaderScope,
    ) -> Result<Vec<Stub>, PgError> {
        if scope.unrestricted() {
            return self.get_related_raw(memory_id, depth, 50);
        }
        if !self.can_see(memory_id, scope)? {
            return Ok(Vec::new());
        }
        let depth = depth.clamp(1, 5);
        let mut seen: HashSet<String> = HashSet::from([memory_id.to_string()]);
        let mut frontier = vec![memory_id.to_string()];
        let mut out: Vec<Stub> = Vec::new();
        for level in 1..=depth {
            let mut next = Vec::new();
            for node in &frontier {
                let rows = self.get_related_raw(node, 1, usize::MAX)?;
                let fresh: Vec<Stub> = rows
                    .into_iter()
                    .filter(|r| {
                        r.id.as_ref()
                            .is_some_and(|i| !i.is_empty() && seen.insert(i.clone()))
                    })
                    .collect();
                let ids: Vec<String> = fresh.iter().filter_map(|s| s.id.clone()).collect();
                let vis = self.visible_ids(&ids, scope)?;
                for mut s in fresh {
                    let id = s.id.clone().unwrap_or_default();
                    if !vis.contains(&id) {
                        continue;
                    }
                    s.distance = Some(level);
                    next.push(id);
                    out.push(s);
                }
            }
            frontier = next;
        }
        out.sort_by(|a, b| {
            (a.distance.unwrap_or(0), a.title.as_deref().unwrap_or(""))
                .cmp(&(b.distance.unwrap_or(0), b.title.as_deref().unwrap_or("")))
        });
        out.truncate(50);
        Ok(out)
    }

    /// Ancestors as `scope` sees them: a hidden start is empty and the chain is
    /// cut at the first hidden ancestor.
    pub fn get_lineage(
        &mut self,
        memory_id: &str,
        scope: &ReaderScope,
    ) -> Result<Vec<Stub>, PgError> {
        if scope.unrestricted() {
            return self.get_lineage_raw(memory_id);
        }
        if !self.can_see(memory_id, scope)? {
            return Ok(Vec::new());
        }
        let chain = self.get_lineage_raw(memory_id)?;
        let ids: Vec<String> = chain.iter().filter_map(|s| s.id.clone()).collect();
        let vis = self.visible_ids(&ids, scope)?;
        let mut out = Vec::new();
        for s in chain {
            if !s.id.as_ref().is_some_and(|i| vis.contains(i)) {
                break;
            }
            out.push(s);
        }
        Ok(out)
    }

    /// The search arm entry point. When the arm is not enabled it returns its
    /// state and no hits without touching the database. Otherwise it is
    /// [`GraphArm::search_by_entity_tokens`].
    pub fn search(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<ArmOutcome, PgError> {
        let state = self.state();
        if state.state != "enabled" {
            return Ok(ArmOutcome {
                state,
                hits: Vec::new(),
            });
        }
        let hits = self.search_by_entity_tokens(query, limit, filters, scope)?;
        Ok(ArmOutcome { state, hits })
    }

    /// Memories mentioning an entity whose name contains `query`
    /// (case-insensitive substring), filters applied before `LIMIT`. With
    /// `as_of` the Cypher over-fetches 3x and the exact validity rule runs on
    /// parsed instants. Capped by the search timeout.
    fn search_by_entity_raw(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
    ) -> Result<Vec<EntityHit>, PgError> {
        if self.graph.is_none() {
            return Err(PgError::BadGraphName(format!("{}_knowledge", self.agent)));
        }
        let q = py_strip(query);
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let as_of = match filters.as_of.as_deref() {
            Some(raw) => normalize_as_of(raw)?,
            None => None,
        };
        let pattern = format!("(?i).*{}.*", py_re_escape(q));
        let (where_, mut params) = Self::memory_filter(filters)?;
        let cyq = format!(
            "MATCH (e:Entity) WHERE e.name =~ $pattern \
             MATCH (m:Memory)-[:MENTIONS]->(e){where_} \
             RETURN DISTINCT m.id, m.title, m.layer, e.name{} LIMIT $limit",
            if as_of.is_some() {
                ", m.valid_from, m.valid_to, m.created_at "
            } else {
                " "
            }
        );
        params.insert("pattern".into(), json!(pattern));
        let fetch = limit * if as_of.is_some() { 3 } else { 1 };
        params.insert("limit".into(), json!(fetch));
        let ncols = if as_of.is_some() { 7 } else { 4 };
        let timeout = Some(self.search_timeout_ms);
        let rows = self.cypher_raw(&cyq, &Value::Object(params), ncols, timeout)?;
        let mut out = Vec::new();
        for r in rows {
            if let Some(t) = as_of.as_deref() {
                let (vf, vt, ca) = (str_of(&r[4]), str_of(&r[5]), str_of(&r[6]));
                if !valid_at(vf.as_deref(), ca.as_deref(), vt.as_deref(), t) {
                    continue;
                }
            }
            out.push(EntityHit {
                id: str_of(&r[0]),
                title: str_of(&r[1]),
                layer: str_of(&r[2]),
                entity: str_of(&r[3]),
            });
        }
        out.truncate(limit);
        Ok(out)
    }

    /// Entity lookup on query TOKENS as well as the whole query: the whole
    /// query first (most specific), then each token of 4+ characters that is
    /// not a stopword, tokens ordered by how few rows they reach and
    /// interleaved by rank so one broad token cannot fill the budget. One time
    /// budget covers the whole fan-out; a sub-lookup that hits its statement
    /// cap abstains (empty) rather than failing the arm.
    pub fn search_by_entity_tokens(
        &mut self,
        query: &str,
        limit: usize,
        filters: &Filters,
        scope: &ReaderScope,
    ) -> Result<Vec<EntityHit>, PgError> {
        if self.graph.is_none() {
            return Err(PgError::BadGraphName(format!("{}_knowledge", self.agent)));
        }
        let q = py_strip(query);
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let budget_s = self.search_timeout_ms.max(0) as f64 / 1000.0;
        let budget_left =
            |started: &Instant| budget_s <= 0.0 || started.elapsed().as_secs_f64() < budget_s;
        let lookup = |arm: &mut GraphArm, text: &str| -> Result<Vec<EntityHit>, PgError> {
            match arm.search_by_entity(text, limit, filters, scope) {
                Err(e) if is_timeout(&e) => Ok(Vec::new()),
                other => other,
            }
        };
        let mut out: Vec<EntityHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for row in lookup(self, query)? {
            let id = row.id.clone().unwrap_or_default();
            if !id.is_empty() && seen.insert(id) {
                out.push(row);
            }
        }
        if out.len() >= limit {
            out.truncate(limit);
            return Ok(out);
        }
        let qkey = q.to_lowercase();
        let mut tokens: Vec<String> = Vec::new();
        for t in find_tokens(query) {
            let low = t.to_lowercase();
            if tokens.contains(&t) || TOKEN_STOPWORDS.contains(&low.as_str()) || low == qkey {
                continue;
            }
            tokens.push(t);
        }
        if tokens.is_empty() {
            return Ok(out);
        }
        let mut per_token: Vec<(usize, String, Vec<EntityHit>)> = Vec::new();
        for tok in tokens {
            if !budget_left(&started) {
                break;
            }
            let rows = lookup(self, &tok)?;
            if !rows.is_empty() {
                per_token.push((rows.len(), tok, rows));
            }
        }
        per_token.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        let mut depth = 0;
        while out.len() < limit && !per_token.is_empty() {
            let mut progressed = false;
            for (_, _, rows) in &per_token {
                if depth >= rows.len() {
                    continue;
                }
                progressed = true;
                let row = &rows[depth];
                let id = row.id.clone().unwrap_or_default();
                if !id.is_empty() && seen.insert(id) {
                    out.push(row.clone());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            if !progressed {
                break;
            }
            depth += 1;
        }
        Ok(out)
    }

    fn resolve_ts(as_of: Option<&str>) -> Result<String, PgError> {
        match as_of {
            Some(raw) => Ok(normalize_as_of(raw)?.unwrap_or_else(now_iso)),
            None => Ok(now_iso()),
        }
    }

    /// IDs whose validity was already CLOSED at `ts`: an incoming `SUPERSEDES`
    /// edge with `valid_to <= ts` as instants. Cypher only drops edges that
    /// cannot qualify (`valid_to` NULL or on a later date).
    fn superseded_ids_as_of(&mut self, ts: &str) -> Result<HashSet<String>, PgError> {
        let t = parse_ts(ts).ok_or_else(|| PgError::BadAsOf(ts.to_string()))?;
        let rows = self.cypher_raw(
            "MATCH (:Memory)-[e:SUPERSEDES]->(p:Memory) \
             WHERE e.valid_to IS NOT NULL AND e.valid_to < $hi \
             RETURN p.id, e.valid_to",
            &json!({ "hi": date_plus(t, 2) }),
            2,
            None,
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let id = str_of(&r[0])?;
                let closed = str_of(&r[1]).and_then(|v| parse_ts(&v))?;
                (closed <= t).then_some(id)
            })
            .collect())
    }

    /// Memories valid at `as_of` (default now): come into being by then
    /// (`created_at` null, unparseable, or `<= as_of`) and not superseded as of
    /// then. Parsed instants throughout. The scan over-fetches 3x `limit`
    /// before the exclusion, in unspecified node order, as Python's `LIMIT`
    /// does.
    fn currently_valid_memories_raw(
        &mut self,
        as_of: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Stub>, PgError> {
        let ts = Self::resolve_ts(as_of)?;
        let t = parse_ts(&ts).ok_or_else(|| PgError::BadAsOf(ts.clone()))?;
        let superseded = self.superseded_ids_as_of(&ts)?;
        let rows = self.cypher_raw(
            "MATCH (m:Memory) \
             WHERE m.created_at IS NULL OR m.created_at < $hi \
             RETURN m.id, m.title, m.layer, m.created_at \
             LIMIT $limit",
            &json!({ "hi": date_plus(t, 2), "limit": limit * 3 }),
            4,
            None,
        )?;
        let mut out = Vec::new();
        for r in rows {
            let created = str_of(&r[3]);
            if created.as_deref().and_then(parse_ts).is_some_and(|c| c > t) {
                continue;
            }
            let id = str_of(&r[0]);
            if id.as_ref().is_some_and(|i| superseded.contains(i)) {
                continue;
            }
            out.push(Stub {
                id,
                title: str_of(&r[1]),
                layer: str_of(&r[2]),
                distance: None,
                created_at: created,
            });
        }
        out.truncate(limit);
        Ok(out)
    }

    /// Whether `memory_id` exists, has come into being by `as_of`, and is not
    /// superseded as of then.
    fn is_currently_valid_raw(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
    ) -> Result<bool, PgError> {
        let ts = Self::resolve_ts(as_of)?;
        let t = parse_ts(&ts).ok_or_else(|| PgError::BadAsOf(ts.clone()))?;
        if self.superseded_ids_as_of(&ts)?.contains(memory_id) {
            return Ok(false);
        }
        let rows = self.cypher_raw(
            "MATCH (m:Memory {id: $id}) RETURN m.created_at",
            &json!({ "id": memory_id }),
            1,
            None,
        )?;
        Ok(rows.iter().any(|r| {
            str_of(&r[0])
                .as_deref()
                .and_then(parse_ts)
                .is_none_or(|c| c <= t)
        }))
    }

    /// Edge window `(valid_from IS NULL OR valid_from <= T) AND (valid_to IS
    /// NULL OR valid_to > T)` on parsed instants. An unparseable bound counts
    /// as absent (always-started / still open).
    fn edge_open_at(valid_from: &Value, valid_to: &Value, t: i128) -> bool {
        let started = str_of(valid_from)
            .as_deref()
            .and_then(parse_ts)
            .is_none_or(|f| f <= t);
        let open = str_of(valid_to)
            .as_deref()
            .and_then(parse_ts)
            .is_none_or(|e| e > t);
        started && open
    }

    /// Direct Memory-to-Memory neighbourhood over an edge valid at `as_of`
    /// (default now), sorted by title, capped at 50. Each of the five edge
    /// types is queried separately (AGE has no relationship alternation).
    fn neighbors_raw(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
        cap: usize,
    ) -> Result<Vec<Stub>, PgError> {
        let ts = Self::resolve_ts(as_of)?;
        let t = parse_ts(&ts).ok_or_else(|| PgError::BadAsOf(ts.clone()))?;
        let mut merged: Vec<Stub> = Vec::new();
        for rel in MEMORY_LINK_EDGES {
            let query = format!(
                "MATCH (start:Memory {{id: $id}})-[e:{rel}]-(n:Memory) \
                 WHERE n.id <> $id \
                 RETURN n.id, n.title, n.layer, e.valid_from, e.valid_to"
            );
            for r in self.cypher_raw(&query, &json!({ "id": memory_id }), 5, None)? {
                if !Self::edge_open_at(&r[3], &r[4], t) {
                    continue;
                }
                let Some(id) = str_of(&r[0]).filter(|i| !i.is_empty()) else {
                    continue;
                };
                if merged.iter().any(|s| s.id.as_deref() == Some(id.as_str())) {
                    continue;
                }
                merged.push(Stub {
                    id: Some(id),
                    title: str_of(&r[1]),
                    layer: str_of(&r[2]),
                    distance: Some(1),
                    created_at: None,
                });
            }
        }
        merged.sort_by(|a, b| {
            a.title
                .as_deref()
                .unwrap_or("")
                .cmp(b.title.as_deref().unwrap_or(""))
        });
        merged.truncate(cap);
        Ok(merged)
    }

    /// The `SUPERSEDES` lineage valid at `as_of` (default now), one hop at a
    /// time with the window applied per hop, nearest first, depth capped at
    /// 10. SUPERSEDES only: promotion and chunk edges never retire a parent.
    fn supersedes_chain_raw(
        &mut self,
        memory_id: &str,
        as_of: Option<&str>,
        visible: &mut dyn FnMut(&mut GraphArm, &str) -> Result<bool, PgError>,
    ) -> Result<Vec<Stub>, PgError> {
        let ts = Self::resolve_ts(as_of)?;
        let t = parse_ts(&ts).ok_or_else(|| PgError::BadAsOf(ts.clone()))?;
        let mut chain: Vec<Stub> = Vec::new();
        let mut visited: HashSet<String> = HashSet::from([memory_id.to_string()]);
        let mut frontier: std::collections::VecDeque<(String, i64)> =
            std::collections::VecDeque::from([(memory_id.to_string(), 0)]);
        while let Some((current, dist)) = frontier.pop_front() {
            if dist >= 10 {
                continue;
            }
            let rows = self.cypher_raw(
                "MATCH (c:Memory {id: $id})-[e:SUPERSEDES]->(p:Memory) \
                 RETURN p.id, p.title, p.layer, e.valid_from, e.valid_to",
                &json!({ "id": current }),
                5,
                None,
            )?;
            for r in rows {
                if !Self::edge_open_at(&r[3], &r[4], t) {
                    continue;
                }
                let Some(pid) = str_of(&r[0]).filter(|i| !i.is_empty()) else {
                    continue;
                };
                if !visited.insert(pid.clone()) {
                    continue;
                }
                // A hidden ancestor is cut: not returned, not walked through.
                if !visible(self, &pid)? {
                    continue;
                }
                chain.push(Stub {
                    id: Some(pid.clone()),
                    title: str_of(&r[1]),
                    layer: str_of(&r[2]),
                    distance: Some(dist + 1),
                    created_at: None,
                });
                frontier.push_back((pid, dist + 1));
            }
        }
        chain.sort_by_key(|s| s.distance);
        Ok(chain)
    }

    /// Memory-to-Memory relatives up to `depth` hops (1 to 5, clamped) over
    /// the five link edges, minimum hop per relative, sorted by distance then
    /// title, capped at 50. Edge validity is not applied (as in Python).
    fn get_related_raw(
        &mut self,
        memory_id: &str,
        depth: i64,
        cap: usize,
    ) -> Result<Vec<Stub>, PgError> {
        let safe = depth.clamp(1, 5);
        let mut merged: HashMap<String, Stub> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for rel in MEMORY_LINK_EDGES {
            let query = format!(
                "MATCH (start:Memory {{id: $id}}) \
                 MATCH path = (start)-[:{rel}*1..{safe}]-(related:Memory) \
                 WHERE related.id <> $id \
                 WITH related, min(length(path)) AS distance \
                 RETURN related.id, related.title, related.layer, distance"
            );
            for r in self.cypher_raw(&query, &json!({ "id": memory_id }), 4, None)? {
                let Some(id) = str_of(&r[0]) else { continue };
                let d = r[3].as_i64().unwrap_or(0);
                match merged.get(&id) {
                    Some(prev) if prev.distance.unwrap_or(0) <= d => {}
                    Some(_) => {
                        merged.get_mut(&id).expect("present").distance = Some(d);
                    }
                    None => {
                        order.push(id.clone());
                        merged.insert(
                            id.clone(),
                            Stub {
                                id: Some(id),
                                title: str_of(&r[1]),
                                layer: str_of(&r[2]),
                                distance: Some(d),
                                created_at: None,
                            },
                        );
                    }
                }
            }
        }
        let mut out: Vec<Stub> = order
            .into_iter()
            .filter_map(|i| merged.remove(&i))
            .collect();
        out.sort_by(|a, b| {
            (a.distance.unwrap_or(0), a.title.as_deref().unwrap_or(""))
                .cmp(&(b.distance.unwrap_or(0), b.title.as_deref().unwrap_or("")))
        });
        out.truncate(cap);
        Ok(out)
    }

    /// Ancestors over `PROMOTED_FROM`, `CHUNK_OF` and `SUPERSEDES` (up to 10
    /// hops), minimum depth per ancestor, nearest first.
    fn get_lineage_raw(&mut self, memory_id: &str) -> Result<Vec<Stub>, PgError> {
        let mut merged: HashMap<String, Stub> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for rel in LINEAGE_WALK_EDGES {
            let query = format!(
                "MATCH (start:Memory {{id: $id}}) \
                 MATCH path = (start)-[:{rel}*1..10]->(anc:Memory) \
                 RETURN anc.id, anc.title, anc.layer, length(path) \
                 ORDER BY length(path) ASC"
            );
            for r in self.cypher_raw(&query, &json!({ "id": memory_id }), 4, None)? {
                let Some(id) = str_of(&r[0]) else { continue };
                let d = r[3].as_i64().unwrap_or(0);
                match merged.get(&id) {
                    Some(prev) if prev.distance.unwrap_or(0) <= d => {}
                    Some(_) => {
                        merged.get_mut(&id).expect("present").distance = Some(d);
                    }
                    None => {
                        order.push(id.clone());
                        merged.insert(
                            id.clone(),
                            Stub {
                                id: Some(id),
                                title: str_of(&r[1]),
                                layer: str_of(&r[2]),
                                distance: Some(d),
                                created_at: None,
                            },
                        );
                    }
                }
            }
        }
        let mut out: Vec<Stub> = order
            .into_iter()
            .filter_map(|i| merged.remove(&i))
            .collect();
        out.sort_by_key(|s| s.distance.unwrap_or(0));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn own() -> OwnerScope {
        OwnerScope::resolve(&crate::env::Env::empty()).unwrap()
    }

    fn owner() -> ReaderScope {
        own().reader_scope().clone()
    }

    #[test]
    fn arm_is_off_unless_asked() {
        let env = |v: Option<&'static str>| move |_: &str| v.map(str::to_string);
        assert!(!graph_arm_enabled(env(None)));
        assert!(!graph_arm_enabled(env(Some("off"))));
        assert!(!graph_arm_enabled(env(Some(""))));
        assert!(!graph_arm_enabled(env(Some("maybe"))));
        for on in ["on", "1", "TRUE", " Yes "] {
            assert!(graph_arm_enabled(env(Some(on))), "{on}");
        }
    }

    #[test]
    fn timeout_env() {
        let env = |v: &'static str| move |_: &str| Some(v.to_string());
        assert_eq!(search_timeout_ms(|_| None), 2500);
        assert_eq!(search_timeout_ms(env("100")), 100);
        assert_eq!(search_timeout_ms(env("0")), 0);
        assert_eq!(search_timeout_ms(env("junk")), 2500);
    }

    #[test]
    fn graph_names_are_identifiers_only() {
        assert!(valid_graph_name("lumina_knowledge"));
        assert!(valid_graph_name("_x9"));
        for bad in [
            "",
            "9x",
            "a-b",
            "a b",
            "x', $$) AS (a agtype); DROP TABLE memories; --",
            "名前",
            "a.b",
        ] {
            assert!(!valid_graph_name(bad), "{bad}");
        }
    }

    #[test]
    fn tokens_match_python_findall() {
        assert_eq!(
            find_tokens("what did Casey say about the invoice"),
            vec!["what", "Casey", "about", "invoice"]
        );
        assert_eq!(find_tokens("ab abc abcd x-y_z.w"), vec!["abcd", "x-y_z.w"]);
        // non-ASCII breaks a token ("caf" is too short, "rich" is found, as findall does)
        assert_eq!(find_tokens("café Zürich"), vec!["rich"]);
        assert_eq!(find_tokens("a-bcde"), vec!["a-bcde"]);
    }

    #[test]
    fn regex_escape_matches_python() {
        assert_eq!(py_re_escape("a.b*c (d) $e"), "a\\.b\\*c\\ \\(d\\)\\ \\$e");
        assert_eq!(py_re_escape("x-y#z&~"), "x\\-y\\#z\\&\\~");
        assert_eq!(py_re_escape("Zürich 東京"), "Zürich\\ 東京");
        assert_eq!(py_re_escape("a\\b"), "a\\\\b");
    }

    #[test]
    fn date_bounds_cover_any_offset() {
        let t = parse_ts("2026-06-01T00:30:00Z").unwrap();
        assert_eq!(date_plus(t, 2), "2026-06-03");
        assert_eq!(date_plus(t, -1), "2026-05-31");
        let t = parse_ts("2026-12-31T23:59:59-05:00").unwrap();
        assert_eq!(date_plus(t, 2), "2027-01-03"); // 04:59Z on 2027-01-01
    }

    #[test]
    fn disabled_arm_reports_and_never_connects() {
        // Nothing listens on port 1: any connection attempt would be an error.
        let cfg: postgres::Config = "host=127.0.0.1 port=1 user=u".parse().unwrap();
        let mut arm = GraphArm::new(cfg, "a", None, false, false).unwrap();
        let st = arm.state();
        assert_eq!(st.state, "disabled");
        assert!(!st.enabled);
        assert_eq!(
            st.reason.as_deref(),
            Some("disabled by config (SKMEMORY_GRAPH_ARM is off, the default)")
        );
        let out = arm
            .search("anything", 5, &Filters::default(), &owner())
            .unwrap();
        assert!(out.hits.is_empty());
        assert_eq!(out.state.state, "disabled");
        // the direct reads are NOT gated by the flag: they try to connect
        let err = arm
            .search_by_entity("x", 5, &Filters::default(), &owner())
            .unwrap_err();
        assert!(matches!(err, PgError::Connect(_)), "{err:?}");
    }

    #[test]
    fn bad_graph_name_makes_the_arm_unavailable() {
        let cfg: postgres::Config = "host=127.0.0.1 port=1 user=u".parse().unwrap();
        let mut arm =
            GraphArm::new(cfg, "a", Some("g'; DROP TABLE memories;--"), true, false).unwrap();
        assert_eq!(arm.state().state, "unavailable");
        assert!(arm.graph().is_none());
        assert!(matches!(
            arm.cypher("RETURN 1", &json!({}), 1, None, &own()),
            Err(PgError::BadGraphName(_))
        ));
        assert!(matches!(
            arm.search("x", 1, &Filters::default(), &owner()),
            Ok(ArmOutcome { hits, .. }) if hits.is_empty()
        ));
    }

    #[test]
    fn remote_host_refused_at_construction() {
        let remote = ["host=", "192.0.2.1", " user=u"].concat();
        let cfg: postgres::Config = remote.parse().unwrap();
        assert!(matches!(
            GraphArm::new(cfg, "a", None, true, false),
            Err(PgError::RemoteDsn { .. })
        ));
    }

    #[test]
    fn filters_become_parameters_never_text() {
        use crate::model::Layer;
        let hostile = "x'] OR 1=1 --$$ RETURN";
        let f = Filters {
            layer: Layer::parse("long-term"),
            tags: vec![hostile.into(), "b".into()],
            context_tag: Some(ContextTag::AnyOf(vec![hostile.into()])),
            source: Some(hostile.into()),
            as_of: Some("2026-06-01T00:00:00Z".into()),
        };
        let (w, p) = GraphArm::memory_filter(&f).unwrap();
        assert!(!w.contains("OR 1=1") && !w.contains("$$") && !w.contains('\''));
        assert_eq!(p["tag0"], hostile);
        assert_eq!(p["context_tags"][0], hostile);
        assert_eq!(p["source"], hostile);
        assert_eq!(p["as_of_hi"], "2026-06-03");
        assert_eq!(p["as_of_lo"], "2026-05-31");
        assert!(w.contains("$tag0 IN m.tags") && w.contains("$tag1 IN m.tags"));
    }
}
