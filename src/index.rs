//! SQLite index over the flat store (SKMEM-R-03): the Rust twin of
//! `skmemory/backends/sqlite_backend.py`, `fts_local.py` and `wal_guard.py`.
//!
//! The live `index.db` is shared with Python nodes during a shadow run, so:
//!
//! * the DDL is the Python text verbatim (identical `sqlite_master` rows);
//! * migrations are idempotent and run in one `BEGIN IMMEDIATE`, so two
//!   openers (Rust or Python) cannot interleave, and a duplicate-column race
//!   is treated as success;
//! * Rust never drops a table, a view or a trigger Python created, and never
//!   deletes an index row whose flat file is missing (see [`Index::reindex`]);
//! * a vaulted store (`fts_enabled == false`) indexes no plaintext and refuses
//!   FTS, exactly as Python's vaulted backend does.

use crate::flat::{json_files, Store};
use crate::model::{Layer, Memory};
use crate::util::{iso_from_unix, now_iso};
use rusqlite::{
    params, params_from_iter, Connection, OptionalExtension, Row, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// `CONTENT_PREVIEW_LENGTH`.
pub const CONTENT_PREVIEW_LENGTH: usize = 150;
/// `REINFORCE_IMPORTANCE_STEP`: `importance += step * (1 - importance)`.
pub const REINFORCE_IMPORTANCE_STEP: f64 = 0.02;
/// Busy wait for ordinary connections (`_BUSY_TIMEOUT_S`).
pub const BUSY_TIMEOUT_MS: u64 = 5000;
/// Busy wait for the hot-path counter bump (`_BUMP_BUSY_MS`).
pub const BUMP_BUSY_MS: u64 = 200;
const BUMP_CHUNK: usize = 500;
/// `wal_guard.JOURNAL_SIZE_LIMIT`.
pub const JOURNAL_SIZE_LIMIT: i64 = 64 * 1024 * 1024;
/// `wal_guard.WAL_AUTOCHECKPOINT`.
pub const WAL_AUTOCHECKPOINT: i64 = 1000;
/// `wal_guard.WAL_DEGRADED_BYTES`.
pub const WAL_DEGRADED_BYTES: u64 = 256 * 1024 * 1024;
/// `wal_guard.CHECKPOINT_INTERVAL_S`.
pub const CHECKPOINT_INTERVAL_S: f64 = 300.0;
const BM25_WEIGHTS: &str = "8.0, 1.0, 3.0, 4.0";
const MAX_TERMS: usize = 32;

/// `_SCHEMA` from sqlite_backend.py, verbatim (comments included: they are
/// part of the stored `CREATE TABLE` text, so parity includes them).
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
id TEXT PRIMARY KEY,
title TEXT NOT NULL,
layer TEXT NOT NULL,
role TEXT NOT NULL DEFAULT 'general',
tags TEXT NOT NULL DEFAULT '',
source TEXT NOT NULL DEFAULT 'manual',
source_ref TEXT NOT NULL DEFAULT '',
summary TEXT NOT NULL DEFAULT '',
content_preview TEXT NOT NULL DEFAULT '',
emotional_intensity REAL NOT NULL DEFAULT 0.0,
emotional_valence REAL NOT NULL DEFAULT 0.0,
emotional_labels TEXT NOT NULL DEFAULT '',
cloud9_achieved INTEGER NOT NULL DEFAULT 0,
importance REAL NOT NULL DEFAULT 0.5,  -- NEW: For prioritization (0.0-1.0)
access_count INTEGER NOT NULL DEFAULT 0,  -- NEW: LRU tracking
last_accessed TEXT,  -- NEW: For expiration/promotion decisions
parent_id TEXT,
related_ids TEXT NOT NULL DEFAULT '',
created_at TEXT NOT NULL,
updated_at TEXT NOT NULL,
file_path TEXT NOT NULL,
content_hash TEXT NOT NULL DEFAULT '',
kind TEXT NOT NULL DEFAULT 'episode',
outcome TEXT,
event_at TEXT,
valid_from TEXT,
valid_to TEXT,
supersedes_id TEXT,
source_tier TEXT NOT NULL DEFAULT 'self-generated',
schema_version INTEGER NOT NULL DEFAULT 1
);

-- Core indexes
CREATE INDEX IF NOT EXISTS idx_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_intensity ON memories(emotional_intensity DESC);
CREATE INDEX IF NOT EXISTS idx_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_parent ON memories(parent_id);

-- NEW: Date-based indexes for lazy loading
CREATE INDEX IF NOT EXISTS idx_date_layer ON memories(DATE(created_at), layer);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_accessed ON memories(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_access_count ON memories(access_count DESC);
CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);

-- Sync failure tracking table
CREATE TABLE IF NOT EXISTS sync_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    error TEXT,
    failed_at TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sync_failures_memory ON sync_failures(memory_id);

-- NEW: View for active context (today + recent summaries)
CREATE VIEW IF NOT EXISTS active_memories AS
SELECT
    id, title, summary, content_preview, tags, layer, created_at,
    importance, access_count,
    CASE
        WHEN DATE(created_at) = CURRENT_DATE THEN 'today'
        WHEN DATE(created_at) = DATE('now', '-1 day') THEN 'yesterday'
        WHEN DATE(created_at) >= DATE('now', '-7 days') THEN 'week'
        ELSE 'historical'
    END as context_tier
FROM memories
WHERE created_at >= DATE('now', '-30 days')
ORDER BY
    context_tier,
    importance DESC,
    access_count DESC;
"#;

/// Columns added by Python's `_init_schema` migration, in its order.
const MIGRATIONS: &[(&str, &str)] = &[
    (
        "importance",
        "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
    ),
    (
        "access_count",
        "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "last_accessed",
        "ALTER TABLE memories ADD COLUMN last_accessed TEXT",
    ),
    (
        "content_hash",
        "ALTER TABLE memories ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
    ),
    (
        "kind",
        "ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'episode'",
    ),
    ("outcome", "ALTER TABLE memories ADD COLUMN outcome TEXT"),
    ("event_at", "ALTER TABLE memories ADD COLUMN event_at TEXT"),
    (
        "valid_from",
        "ALTER TABLE memories ADD COLUMN valid_from TEXT",
    ),
    ("valid_to", "ALTER TABLE memories ADD COLUMN valid_to TEXT"),
    (
        "supersedes_id",
        "ALTER TABLE memories ADD COLUMN supersedes_id TEXT",
    ),
    (
        "source_tier",
        "ALTER TABLE memories ADD COLUMN source_tier TEXT NOT NULL DEFAULT 'self-generated'",
    ),
    (
        "schema_version",
        "ALTER TABLE memories ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1",
    ),
];

/// `fts_local._DDL`, verbatim.
const FTS_DDL: &[&str] = &[
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(\
title, content, summary, tags, content='', contentless_delete=1, \
tokenize='unicode61 remove_diacritics 2')",
    "CREATE TABLE IF NOT EXISTS fts_meta (\
rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, \
content_hash TEXT NOT NULL DEFAULT '', context_tag TEXT NOT NULL DEFAULT '')",
    "CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN \
DELETE FROM memory_fts WHERE rowid IN (SELECT rowid FROM fts_meta WHERE id = old.id); \
DELETE FROM fts_meta WHERE id = old.id; END",
];

/// Errors from the index.
#[derive(Debug)]
pub enum IndexError {
    Sqlite(rusqlite::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Sqlite(e) => write!(f, "sqlite: {e}"),
            IndexError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<rusqlite::Error> for IndexError {
    fn from(e: rusqlite::Error) -> Self {
        IndexError::Sqlite(e)
    }
}

impl From<std::io::Error> for IndexError {
    fn from(e: std::io::Error) -> Self {
        IndexError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, IndexError>;

/// `decay.initial_importance`: layer base + 0.3 * intensity/10, capped at 1.0,
/// rounded to 4 places (Python `round`, half-even on the exact value, which is
/// what Rust's fixed-precision formatting does too).
pub fn initial_importance(layer: &str, intensity: f64) -> f64 {
    let base = match layer {
        "short-term" => 0.3,
        "mid-term" => 0.5,
        "long-term" => 0.7,
        _ => 0.5,
    };
    let i = if intensity.is_nan() {
        0.0
    } else {
        intensity.clamp(0.0, 10.0)
    } / 10.0;
    let v = base + 0.3 * i;
    let rounded: f64 = format!("{v:.4}").parse().unwrap_or(v);
    rounded.min(1.0)
}

/// `_is_legacy_importance`: a stored 0.5 with no reinforcement is the
/// pre-P0-03 column default, so it is safe to recompute.
pub fn is_legacy_importance(importance: Option<f64>, access_count: i64) -> bool {
    matches!(importance, Some(v) if (v - 0.5).abs() < 1e-9) && access_count == 0
}

/// `Memory.content_hash()`: first 16 hex chars of SHA-256 over the content.
pub fn content_hash(content: &str) -> String {
    let d = Sha256::digest(content.as_bytes());
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn word_chars(chunk: &str) -> Vec<String> {
    // Python `\w+` (Unicode): alphanumerics and underscore.
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in chunk.chars() {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `fts_local.sanitize_match`: free text to a MATCH expression that cannot
/// raise a syntax error (every chunk a quoted phrase, joined by `joiner`).
pub fn sanitize_match(query: &str, joiner: &str) -> Option<String> {
    let mut terms = Vec::new();
    for chunk in query.split_whitespace() {
        let parts = word_chars(chunk);
        if !parts.is_empty() {
            terms.push(format!("\"{}\"", parts.join(" ")));
        }
        if terms.len() >= MAX_TERMS {
            break;
        }
    }
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(joiner))
    }
}

fn is_corruption(e: &rusqlite::Error) -> bool {
    let m = e.to_string().to_lowercase();
    m.contains("malformed")
        || m.contains("not a database")
        || m.contains("disk image")
        || m.contains("file is encrypted")
        || m.contains("database corruption")
}

static LAST_CHECKPOINT: Mutex<Option<HashMap<String, Instant>>> = Mutex::new(None);

/// `wal_guard.apply`: bound the WAL. Never fails.
fn wal_apply(conn: &Connection) {
    let _ = conn.query_row(
        &format!("PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT}"),
        [],
        |_| Ok(()),
    );
    let _ = conn.execute_batch(&format!("PRAGMA wal_autocheckpoint={WAL_AUTOCHECKPOINT}"));
}

/// `wal_guard.maybe_checkpoint`: throttled PASSIVE checkpoint, which never
/// blocks a reader or a writer. Returns whether one ran.
pub fn maybe_checkpoint(conn: &Connection, key: &str, interval_s: f64) -> bool {
    {
        let mut g = LAST_CHECKPOINT.lock().unwrap_or_else(|p| p.into_inner());
        let map = g.get_or_insert_with(HashMap::new);
        let now = Instant::now();
        if let Some(t) = map.get(key) {
            if now.duration_since(*t).as_secs_f64() < interval_s {
                return false;
            }
        }
        map.insert(key.to_string(), now);
    }
    conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))
        .is_ok()
}

/// `wal_guard.wal_status` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalStatus {
    pub wal_size_bytes: u64,
    pub wal_degraded_threshold_bytes: u64,
    pub wal_ok: bool,
}

/// On-disk `-wal` size and whether it exceeds `degraded_bytes`.
pub fn wal_status(db_path: &Path, degraded_bytes: u64) -> WalStatus {
    let mut p = db_path.as_os_str().to_owned();
    p.push("-wal");
    let size = std::fs::metadata(PathBuf::from(p))
        .map(|m| m.len())
        .unwrap_or(0);
    WalStatus {
        wal_size_bytes: size,
        wal_degraded_threshold_bytes: degraded_bytes,
        wal_ok: size <= degraded_bytes,
    }
}

/// One `memories` row, every column (no rowid).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRow {
    pub id: String,
    pub title: String,
    pub layer: String,
    pub role: String,
    pub tags: String,
    pub source: String,
    pub source_ref: String,
    pub summary: String,
    pub content_preview: String,
    pub emotional_intensity: f64,
    pub emotional_valence: f64,
    pub emotional_labels: String,
    pub cloud9_achieved: i64,
    pub importance: f64,
    pub access_count: i64,
    pub last_accessed: Option<String>,
    pub parent_id: Option<String>,
    pub related_ids: String,
    pub created_at: String,
    pub updated_at: String,
    pub file_path: String,
    pub content_hash: String,
    pub kind: String,
    pub outcome: Option<String>,
    pub event_at: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub supersedes_id: Option<String>,
    pub source_tier: String,
    pub schema_version: i64,
}

impl MemoryRow {
    fn from_row(r: &Row<'_>) -> rusqlite::Result<MemoryRow> {
        Ok(MemoryRow {
            id: r.get("id")?,
            title: r.get("title")?,
            layer: r.get("layer")?,
            role: r.get("role")?,
            tags: r.get("tags")?,
            source: r.get("source")?,
            source_ref: r.get("source_ref")?,
            summary: r.get("summary")?,
            content_preview: r.get("content_preview")?,
            emotional_intensity: r.get("emotional_intensity")?,
            emotional_valence: r.get("emotional_valence")?,
            emotional_labels: r.get("emotional_labels")?,
            cloud9_achieved: r.get("cloud9_achieved")?,
            importance: r.get("importance")?,
            access_count: r.get("access_count")?,
            last_accessed: r.get("last_accessed")?,
            parent_id: r.get("parent_id")?,
            related_ids: r.get("related_ids")?,
            created_at: r.get("created_at")?,
            updated_at: r.get("updated_at")?,
            file_path: r.get("file_path")?,
            content_hash: r.get("content_hash")?,
            kind: r.get("kind")?,
            outcome: r.get("outcome")?,
            event_at: r.get("event_at")?,
            valid_from: r.get("valid_from")?,
            valid_to: r.get("valid_to")?,
            supersedes_id: r.get("supersedes_id")?,
            source_tier: r.get("source_tier")?,
            schema_version: r.get("schema_version")?,
        })
    }
}

/// Filters for [`Index::search_lexical`] (`fts_local.search_rows`).
#[derive(Debug, Clone, Default)]
pub struct SearchFilters<'a> {
    pub layer: Option<&'a str>,
    /// AND semantics.
    pub tags: &'a [String],
    pub source: Option<&'a str>,
    /// `None` is no filter; `Some(&[])` matches nothing (fails closed).
    pub context_tags: Option<&'a [String]>,
    /// Already-normalised UTC ISO instant (see Python `validity.normalize_as_of`).
    pub as_of: Option<&'a str>,
}

/// Result of [`Index::reindex`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReindexReport {
    /// Flat files indexed.
    pub indexed: usize,
    /// Flat files that could not be parsed (skipped, never fatal).
    pub unloadable: usize,
    /// Index rows with no flat file that were kept (not dropped).
    pub orphans_kept: usize,
    /// Rows whose usage state was carried across the rebuild.
    pub usage_restored: usize,
}

/// Result of [`Index::backfill_importance`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportanceBackfill {
    pub total: i64,
    pub scanned: i64,
    pub updated: i64,
    pub unchanged: i64,
}

/// FTS coverage (`fts_local.coverage`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    pub total: i64,
    pub indexed: i64,
    pub missing: i64,
}

/// Result of [`Index::fts_backfill`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackfillReport {
    pub indexed: usize,
    pub unloadable: usize,
    pub complete: bool,
}

/// The SQLite index over a flat store's `index.db`.
pub struct Index {
    path: PathBuf,
    conn: Connection,
    fts_enabled: bool,
    fts_ok: bool,
}

fn open_conn(path: &Path, busy_ms: u64) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_millis(busy_ms))?;
    // journal_mode returns a row; a concurrent opener may already have set WAL.
    let _ = conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()));
    conn.execute_batch("PRAGMA synchronous=NORMAL")?;
    wal_apply(&conn);
    Ok(conn)
}

impl Index {
    /// Open `<root>/index.db`, creating or migrating the schema (idempotent,
    /// safe against a concurrent Python or Rust opener). `fts_enabled = false`
    /// is the vaulted mode: FTS artefacts are dropped and search refuses.
    pub fn open(root: impl AsRef<Path>, fts_enabled: bool) -> Result<Index> {
        Index::open_db(root.as_ref().join("index.db"), fts_enabled)
    }

    /// As [`Index::open`] with an explicit database path.
    pub fn open_db(path: impl Into<PathBuf>, fts_enabled: bool) -> Result<Index> {
        let path = path.into();
        let conn = open_conn(&path, BUSY_TIMEOUT_MS)?;
        let mut idx = Index {
            path,
            conn,
            fts_enabled,
            fts_ok: false,
        };
        idx.init_schema()?;
        Ok(idx)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fts_enabled(&self) -> bool {
        self.fts_enabled
    }

    /// Whether the FTS arm is usable (enabled and FTS5 available).
    pub fn fts_ok(&self) -> bool {
        self.fts_ok
    }

    /// Direct access for read-only inspection and tests.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn init_schema(&mut self) -> Result<()> {
        // One immediate transaction: the column checks and ALTERs cannot
        // interleave with another opener's. Nothing is ever dropped (Python
        // drops and recreates `active_memories`; Rust leaves a node's view be).
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let has_table: bool = tx
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if has_table {
            let mut existing = BTreeSet::new();
            {
                let mut st = tx.prepare("PRAGMA table_info(memories)")?;
                let rows = st.query_map([], |r| r.get::<_, String>(1))?;
                for r in rows {
                    existing.insert(r?);
                }
            }
            for (col, ddl) in MIGRATIONS {
                if !existing.contains(*col) {
                    if let Err(e) = tx.execute_batch(ddl) {
                        if !e.to_string().to_lowercase().contains("duplicate column") {
                            return Err(e.into());
                        }
                    }
                }
            }
        }
        tx.execute_batch(SCHEMA)?;
        tx.commit()?;

        if self.fts_enabled {
            self.fts_ok = self.fts_ensure_schema();
        } else {
            self.fts_ok = false;
            self.fts_drop()?;
        }
        Ok(())
    }

    fn fts_ensure_schema(&self) -> bool {
        let run = || -> rusqlite::Result<()> {
            let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
            for ddl in FTS_DDL {
                tx.execute_batch(ddl)?;
            }
            tx.commit()
        };
        run().is_ok()
    }

    /// Vaulted mode: remove every FTS artefact, then VACUUM so the dropped
    /// pages (which still hold plaintext) are scrubbed. The drop is what
    /// Python's vaulted backend does too.
    fn fts_drop(&self) -> Result<()> {
        let existed: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE name LIKE 'memory_fts%' OR name = 'fts_meta' LIMIT 1",
                [],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        self.conn.execute_batch(
            "DROP TRIGGER IF EXISTS memories_fts_ad; \
             DROP TABLE IF EXISTS memory_fts; \
             DROP TABLE IF EXISTS fts_meta;",
        )?;
        if existed {
            // A locked database must not make opening the vault fail.
            let _ = self.conn.execute_batch("VACUUM");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // writes
    // ------------------------------------------------------------------

    /// Insert or update the index entry (`_index_memory`). The UPSERT rewrites
    /// content columns only: `access_count`, `last_accessed` and `importance`
    /// are usage state and survive a re-save. Importance is set on first
    /// insert only. Commits and runs the throttled PASSIVE checkpoint.
    pub fn index_memory(&self, memory: &Memory, file_path: &Path) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        self.upsert_in(&tx, memory, file_path)?;
        tx.commit()?;
        maybe_checkpoint(
            &self.conn,
            &self.path.to_string_lossy(),
            CHECKPOINT_INTERVAL_S,
        );
        Ok(())
    }

    /// The UPSERT plus the FTS row, inside the caller's transaction.
    fn upsert_in(&self, conn: &Connection, m: &Memory, file_path: &Path) -> Result<()> {
        let preview: String = m.content.chars().take(CONTENT_PREVIEW_LENGTH).collect();
        let hash = content_hash(&m.content);
        let mut st = conn.prepare_cached(UPSERT_SQL)?;
        st.execute(params![
            m.id,
            m.title,
            m.layer.as_str(),
            m.role.as_str(),
            m.tags.join(","),
            m.source,
            m.source_ref,
            m.summary,
            preview,
            m.emotional.intensity,
            m.emotional.valence,
            m.emotional.labels.join(","),
            i64::from(m.emotional.cloud9_achieved),
            m.parent_id,
            m.related_ids.join(","),
            m.created_at,
            m.updated_at,
            file_path.to_string_lossy(),
            hash,
            m.kind,
            m.outcome,
            m.event_at,
            m.valid_from,
            m.valid_to,
            m.supersedes_id,
            m.source_tier,
            m.schema_version,
            initial_importance(m.layer.as_str(), m.emotional.intensity),
        ])?;
        if self.fts_ok {
            // The FTS arm is derived; a failure must not fail the index write.
            let _ = self.fts_index_in(conn, m, &hash);
        }
        Ok(())
    }

    fn fts_index_in(&self, conn: &Connection, m: &Memory, hash: &str) -> rusqlite::Result<()> {
        let existing: Option<i64> = conn
            .query_row("SELECT rowid FROM fts_meta WHERE id = ?", [&m.id], |r| {
                r.get(0)
            })
            .optional()?;
        let rid = match existing {
            None => {
                conn.execute(
                    "INSERT INTO fts_meta(id, content_hash, context_tag) VALUES (?, ?, ?)",
                    params![m.id, hash, m.context_tag],
                )?;
                conn.last_insert_rowid()
            }
            Some(rid) => {
                conn.execute(
                    "UPDATE fts_meta SET content_hash = ?, context_tag = ? WHERE rowid = ?",
                    params![hash, m.context_tag, rid],
                )?;
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", [rid])?;
                rid
            }
        };
        conn.execute(
            "INSERT INTO memory_fts(rowid, title, content, summary, tags) VALUES (?, ?, ?, ?, ?)",
            params![rid, m.title, m.content, m.summary, m.tags.join(" ")],
        )?;
        Ok(())
    }

    /// Remove an index row (the AFTER DELETE trigger clears the FTS rows).
    pub fn delete(&self, id: &str) -> Result<bool> {
        Ok(self
            .conn
            .execute("DELETE FROM memories WHERE id = ?", [id])?
            > 0)
    }

    // ------------------------------------------------------------------
    // reads
    // ------------------------------------------------------------------

    /// One row by id.
    pub fn get(&self, id: &str) -> Result<Option<MemoryRow>> {
        Ok(self
            .conn
            .query_row("SELECT * FROM memories WHERE id = ?", [id], |r| {
                MemoryRow::from_row(r)
            })
            .optional()?)
    }

    /// Every row ordered by id.
    pub fn all_rows(&self) -> Result<Vec<MemoryRow>> {
        let mut st = self.conn.prepare("SELECT * FROM memories ORDER BY id")?;
        let rows = st.query_map([], |r| MemoryRow::from_row(r))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
    }

    // ------------------------------------------------------------------
    // usage counters
    // ------------------------------------------------------------------

    /// Reinforce memories a caller read (`bump_access`): one batched UPDATE per
    /// 500 ids setting `access_count + 1`, `importance += 0.02 * (1 - importance)`
    /// and `last_accessed`. With `debounce_s`, a row is bumped only when
    /// `last_accessed` is NULL or older than that. Runs on its OWN short-lived
    /// connection with a 200 ms busy timeout and NEVER fails: under contention
    /// or a broken index the bump is skipped and 0 returned.
    pub fn bump_access(&self, ids: &[&str], now: Option<&str>, debounce_s: Option<f64>) -> usize {
        let mut seen = BTreeSet::new();
        let ids: Vec<&str> = ids
            .iter()
            .copied()
            .filter(|i| !i.is_empty() && seen.insert(*i))
            .collect();
        if ids.is_empty() {
            return 0;
        }
        let stamp = now.map(str::to_string).unwrap_or_else(now_iso);
        let (guard, cutoff) = match debounce_s {
            Some(d) => (
                " AND (last_accessed IS NULL OR last_accessed < ?)",
                Some(cutoff_iso(d)),
            ),
            None => ("", None),
        };
        let run = || -> rusqlite::Result<usize> {
            let mut conn = Connection::open(&self.path)?;
            conn.busy_timeout(Duration::from_millis(BUMP_BUSY_MS))?;
            wal_apply(&conn);
            let tx = conn.transaction()?;
            let mut updated = 0;
            for chunk in ids.chunks(BUMP_CHUNK) {
                let marks = vec!["?"; chunk.len()].join(",");
                let sql = format!(
                    "UPDATE memories SET access_count = access_count + 1, \
                     importance = importance + {REINFORCE_IMPORTANCE_STEP} * (1.0 - importance), \
                     last_accessed = ? WHERE id IN ({marks}){guard}"
                );
                let mut args: Vec<&str> = vec![stamp.as_str()];
                args.extend(chunk.iter().copied());
                if let Some(c) = &cutoff {
                    args.push(c.as_str());
                }
                updated += tx.execute(&sql, params_from_iter(args))?;
            }
            tx.commit()?;
            Ok(updated)
        };
        run().unwrap_or(0)
    }

    /// `(access_count, last_accessed)` per id; unindexed ids are absent.
    pub fn get_usage(&self, ids: &[&str]) -> HashMap<String, (i64, Option<String>)> {
        let mut out = HashMap::new();
        let mut seen = BTreeSet::new();
        let ids: Vec<&str> = ids
            .iter()
            .copied()
            .filter(|i| !i.is_empty() && seen.insert(*i))
            .collect();
        for chunk in ids.chunks(BUMP_CHUNK) {
            let marks = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT id, access_count, last_accessed FROM memories WHERE id IN ({marks})"
            );
            let Ok(mut st) = self.conn.prepare(&sql) else {
                return out;
            };
            let Ok(rows) = st.query_map(params_from_iter(chunk.iter()), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    r.get::<_, Option<String>>(2)?,
                ))
            }) else {
                return out;
            };
            for r in rows.flatten() {
                out.insert(r.0, (r.1, r.2));
            }
        }
        out
    }

    /// Recompute legacy importance (exactly 0.5 with no accesses) from layer
    /// and intensity, index only, in small batches with a commit each.
    pub fn backfill_importance(&self, batch_size: usize) -> Result<ImportanceBackfill> {
        let total = self.count()?;
        let batch = batch_size.max(1) as i64;
        let (mut scanned, mut updated) = (0i64, 0i64);
        let mut last = String::new();
        loop {
            let rows: Vec<(String, String, f64)> = {
                let mut st = self.conn.prepare(
                    "SELECT id, layer, emotional_intensity FROM memories \
                     WHERE id > ? AND importance = 0.5 AND access_count = 0 ORDER BY id LIMIT ?",
                )?;
                let it = st.query_map(params![last, batch], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?;
                it.collect::<rusqlite::Result<_>>()?
            };
            let Some(tail) = rows.last() else { break };
            last = tail.0.clone();
            let changes: Vec<(f64, &str)> = rows
                .iter()
                .filter_map(|(id, layer, inten)| {
                    let new = initial_importance(layer, *inten);
                    ((new - 0.5).abs() > 1e-9).then_some((new, id.as_str()))
                })
                .collect();
            scanned += rows.len() as i64;
            if !changes.is_empty() {
                let tx = self.conn.unchecked_transaction()?;
                for (v, id) in &changes {
                    tx.execute(
                        "UPDATE memories SET importance = ? WHERE id = ? \
                         AND importance = 0.5 AND access_count = 0",
                        params![v, id],
                    )?;
                }
                tx.commit()?;
                updated += changes.len() as i64;
            }
        }
        Ok(ImportanceBackfill {
            total,
            scanned,
            updated,
            unchanged: scanned - updated,
        })
    }

    // ------------------------------------------------------------------
    // reindex
    // ------------------------------------------------------------------

    /// Rebuild the index from the flat files, usage state preserved.
    ///
    /// Snapshot, clear, re-index and restore all happen in ONE
    /// `BEGIN IMMEDIATE` transaction: a Python reader never sees an empty or
    /// half-built index and no bump can land between snapshot and restore
    /// (Python does it in two commits). The restore is additive for the count
    /// and max for the timestamp, and a snapshotted importance of exactly 0.5
    /// with no accesses is the legacy default, so the freshly computed value
    /// stands (`_is_legacy_importance`).
    ///
    /// Deviation from Python, deliberate: Python's default first writes
    /// SQLite-only rows out to flat files. Rust never writes flat files from
    /// the index, so unless `force` an index row with no flat file is KEPT
    /// (tombstoned ids excepted, which Python also refuses to resurrect).
    /// `force` drops them, like Python's `force=True`.
    ///
    /// Holds the write lock for the whole rebuild: concurrent Python writers
    /// wait up to their 5 s busy timeout, then skip their index write (the
    /// flat file is intact and the next reindex picks it up).
    pub fn reindex(&self, store: &Store, force: bool) -> Result<ReindexReport> {
        let flat_ids = store.flat_id_set();
        let mut report = ReindexReport::default();

        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;

        let snapshot: Vec<(String, i64, Option<String>, f64)> = {
            let mut st =
                tx.prepare("SELECT id, access_count, last_accessed, importance FROM memories")?;
            let it = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
            it.collect::<rusqlite::Result<_>>()?
        };
        let keep: BTreeSet<&str> = if force {
            BTreeSet::new()
        } else {
            snapshot
                .iter()
                .map(|s| s.0.as_str())
                .filter(|id| !flat_ids.contains(*id) && !store.is_tombstoned(id))
                .collect()
        };
        report.orphans_kept = keep.len();
        if keep.is_empty() {
            tx.execute_batch("DELETE FROM memories")?;
        } else {
            tx.execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS _reindex_keep (id TEXT PRIMARY KEY); \
                 DELETE FROM _reindex_keep",
            )?;
            {
                let mut ins = tx.prepare("INSERT OR IGNORE INTO _reindex_keep (id) VALUES (?)")?;
                for id in &keep {
                    ins.execute([id])?;
                }
            }
            tx.execute_batch(
                "DELETE FROM memories WHERE id NOT IN (SELECT id FROM _reindex_keep); \
                 DROP TABLE _reindex_keep",
            )?;
        }

        for layer in Layer::ALL {
            for path in json_files(&store.layer_dir(layer)) {
                if store.is_hidden_conflict_copy(&path) {
                    continue; // a stale conflict copy must not overwrite the canonical row
                }
                match store.load_path(&path) {
                    Ok(m) => {
                        self.upsert_in(&tx, &m, &path)?;
                        report.indexed += 1;
                    }
                    Err(_) => report.unloadable += 1,
                }
            }
        }

        {
            let mut up = tx.prepare(
                "UPDATE memories SET access_count = access_count + ?, \
                 last_accessed = CASE WHEN last_accessed IS NULL OR last_accessed < ? \
                 THEN ? ELSE last_accessed END, importance = COALESCE(?, importance) \
                 WHERE id = ?",
            )?;
            for (id, count, last, imp) in &snapshot {
                if keep.contains(id.as_str()) {
                    continue; // never deleted, nothing to restore
                }
                let keep_imp = if is_legacy_importance(Some(*imp), *count) {
                    None
                } else {
                    Some(*imp)
                };
                report.usage_restored += up.execute(params![
                    count,
                    last.as_deref().unwrap_or(""),
                    last,
                    keep_imp,
                    id
                ])?;
            }
        }
        tx.commit()?;
        maybe_checkpoint(
            &self.conn,
            &self.path.to_string_lossy(),
            CHECKPOINT_INTERVAL_S,
        );
        Ok(report)
    }

    // ------------------------------------------------------------------
    // FTS
    // ------------------------------------------------------------------

    /// Indexed vs total (`fts_local.coverage`); `None` when FTS is off.
    pub fn fts_status(&self) -> Result<Option<Coverage>> {
        if !self.fts_ok {
            return Ok(None);
        }
        let total = self.count()?;
        let indexed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories m JOIN fts_meta f ON f.id = m.id",
            [],
            |r| r.get(0),
        )?;
        Ok(Some(Coverage {
            total,
            indexed,
            missing: total - indexed,
        }))
    }

    /// Index rows that have no FTS entry (`fts_local.backfill`), loading each
    /// from its flat file. An unloadable row gets an `'unloadable'` marker so
    /// it is not retried. No-op when FTS is off (vaulted).
    pub fn fts_backfill(
        &self,
        store: &Store,
        max_rows: Option<usize>,
        budget: Option<Duration>,
    ) -> Result<BackfillReport> {
        if !self.fts_ok {
            return Ok(BackfillReport::default());
        }
        let start = Instant::now();
        let (mut done, mut skipped) = (0usize, 0usize);
        loop {
            let mut batch = 200usize;
            if let Some(m) = max_rows {
                batch = batch.min(m.saturating_sub(done + skipped));
                if batch == 0 {
                    break;
                }
            }
            let rows: Vec<(String, String, String)> = {
                let mut st = self.conn.prepare(
                    "SELECT m.id, m.file_path, m.content_hash FROM memories m \
                     LEFT JOIN fts_meta f ON f.id = m.id WHERE f.id IS NULL LIMIT ?",
                )?;
                let it = st.query_map([batch as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
                it.collect::<rusqlite::Result<_>>()?
            };
            if rows.is_empty() {
                break;
            }
            let tx = self.conn.unchecked_transaction()?;
            for (id, fp, hash) in &rows {
                match store.load_path(Path::new(fp)) {
                    Ok(m) => {
                        self.fts_index_in(&tx, &m, hash)?;
                        done += 1;
                    }
                    Err(_) => {
                        tx.execute(
                            "INSERT OR REPLACE INTO fts_meta(id, content_hash) VALUES (?, 'unloadable')",
                            [id],
                        )?;
                        skipped += 1;
                    }
                }
                if budget.is_some_and(|b| start.elapsed() >= b) {
                    tx.commit()?;
                    return Ok(BackfillReport {
                        indexed: done,
                        unloadable: skipped,
                        complete: false,
                    });
                }
            }
            tx.commit()?;
        }
        let left: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories m LEFT JOIN fts_meta f ON f.id = m.id WHERE f.id IS NULL",
            [],
            |r| r.get(0),
        )?;
        Ok(BackfillReport {
            indexed: done,
            unloadable: skipped,
            complete: left == 0,
        })
    }

    /// bm25-ranked full-text search over title, content, summary and tags
    /// (`search_rows`): all terms first (AND), any term as the fallback (OR).
    /// Returns nothing when FTS is off, as the vaulted Python backend does.
    pub fn search_lexical(
        &self,
        query: &str,
        limit: usize,
        f: &SearchFilters<'_>,
    ) -> Result<Vec<MemoryRow>> {
        if !self.fts_ok || limit == 0 {
            return Ok(Vec::new());
        }
        let Some(match_or) = sanitize_match(query, " OR ") else {
            return Ok(Vec::new());
        };
        let mut wh = vec!["memory_fts MATCH ?".to_string()];
        let mut args: Vec<String> = vec![String::new()];
        if let Some(l) = f.layer {
            wh.push("m.layer = ?".into());
            args.push(l.to_string());
        }
        for t in f.tags {
            wh.push("instr(',' || m.tags || ',', ?) > 0".into());
            args.push(format!(",{t},"));
        }
        if let Some(s) = f.source.filter(|s| !s.is_empty()) {
            wh.push("m.source = ?".into());
            args.push(s.to_string());
        }
        if let Some(ctx) = f.context_tags {
            if ctx.is_empty() {
                return Ok(Vec::new());
            }
            wh.push(format!(
                "f.context_tag IN ({})",
                vec!["?"; ctx.len()].join(",")
            ));
            args.extend(ctx.iter().cloned());
        }
        if let Some(a) = f.as_of.filter(|a| !a.is_empty()) {
            wh.push(
                "COALESCE(julianday(m.valid_from), julianday(m.created_at), 0) <= julianday(?) \
                 AND (julianday(m.valid_to) IS NULL OR julianday(m.valid_to) > julianday(?))"
                    .into(),
            );
            args.push(a.to_string());
            args.push(a.to_string());
        }
        let sql = format!(
            "SELECT m.* FROM memory_fts \
             JOIN fts_meta f ON f.rowid = memory_fts.rowid \
             JOIN memories m ON m.id = f.id \
             WHERE {} ORDER BY bm25(memory_fts, {BM25_WEIGHTS}) LIMIT ?",
            wh.join(" AND ")
        );
        let and = sanitize_match(query, " AND ");
        for m in [and.as_deref(), Some(match_or.as_str())]
            .into_iter()
            .flatten()
        {
            args[0] = m.to_string();
            let mut st = self.conn.prepare(&sql)?;
            let mut all: Vec<rusqlite::types::Value> = args
                .iter()
                .cloned()
                .map(rusqlite::types::Value::from)
                .collect();
            all.push(rusqlite::types::Value::from(limit as i64));
            let rows: Vec<MemoryRow> = st
                .query_map(params_from_iter(all), |r| MemoryRow::from_row(r))?
                .collect::<rusqlite::Result<_>>()?;
            if !rows.is_empty() || m == match_or {
                return Ok(rows);
            }
        }
        Ok(Vec::new())
    }

    /// True when `err` means the index file itself is damaged (the caller may
    /// quarantine it; this crate never moves or deletes an index file).
    pub fn is_corruption(err: &IndexError) -> bool {
        matches!(err, IndexError::Sqlite(e) if is_corruption(e))
    }

    /// `wal_status` for this database.
    pub fn wal_status(&self) -> WalStatus {
        wal_status(&self.path, WAL_DEGRADED_BYTES)
    }

    /// Run a PASSIVE checkpoint now if the throttle allows.
    pub fn maybe_checkpoint(&self) -> bool {
        maybe_checkpoint(
            &self.conn,
            &self.path.to_string_lossy(),
            CHECKPOINT_INTERVAL_S,
        )
    }
}

/// `(now - debounce_s).isoformat()`.
fn cutoff_iso(debounce_s: f64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let micros = now.as_micros() as i128 - (debounce_s * 1e6).round() as i128;
    let secs = micros.div_euclid(1_000_000) as i64;
    let sub = micros.rem_euclid(1_000_000) as u32;
    iso_from_unix(secs, sub)
}

const UPSERT_SQL: &str = "
INSERT INTO memories (
    id, title, layer, role, tags, source, source_ref,
    summary, content_preview, emotional_intensity,
    emotional_valence, emotional_labels, cloud9_achieved,
    parent_id, related_ids, created_at, updated_at,
    file_path, content_hash,
    kind, outcome, event_at, valid_from, valid_to,
    supersedes_id, source_tier, schema_version, importance
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    title = excluded.title,
    layer = excluded.layer,
    role = excluded.role,
    tags = excluded.tags,
    source = excluded.source,
    source_ref = excluded.source_ref,
    summary = excluded.summary,
    content_preview = excluded.content_preview,
    emotional_intensity = excluded.emotional_intensity,
    emotional_valence = excluded.emotional_valence,
    emotional_labels = excluded.emotional_labels,
    cloud9_achieved = excluded.cloud9_achieved,
    parent_id = excluded.parent_id,
    related_ids = excluded.related_ids,
    created_at = excluded.created_at,
    updated_at = excluded.updated_at,
    file_path = excluded.file_path,
    content_hash = excluded.content_hash,
    kind = excluded.kind,
    outcome = excluded.outcome,
    event_at = excluded.event_at,
    valid_from = excluded.valid_from,
    valid_to = excluded.valid_to,
    supersedes_id = excluded.supersedes_id,
    source_tier = excluded.source_tier,
    schema_version = excluded.schema_version
";
