//! Write-time behaviour of `MemoryStore.snapshot` that the Rust engine reproduces
//! (SKMEM-R-07C part 2): the content-hash dedup guard, the source tier derived
//! from `source`, the authority tier written into `metadata`, and the valence
//! inferred from emotion labels. Each is checked against the real Python
//! `snapshot` in `tests/snapshot_parity.rs` (vectors from
//! `tests/snapshot_parity/gen_snapshot_parity.py`).
//!
//! What stays in Python (the shim keeps calling it) is listed in
//! `docs/PYTHON_BINDINGS.md` under "snapshot side effects": document
//! decomposition and splitting (content of 1200 characters or more, or
//! `decompose=True`, is REFUSED here, never stored whole), write-time triage
//! (`kind` fact, rule, preference, and the closing of a superseded head), the
//! pgvector row, the graph upsert, the write-ahead log, custom pre-write hooks,
//! reweave and the SKSeed truth check.

use crate::audience::OwnerScope;
use crate::engine::{Engine, EngineError};
use crate::index::content_hash;
use crate::model::{py_strip, Memory, ModelError};
use serde_json::{Map, Value};

/// `DECOMPOSE_MIN_LENGTH`: content this long (in characters) is decomposed by
/// Python's `ingest_document`, which this engine does not do.
pub const DECOMPOSE_MIN_LENGTH: usize = 1200;

/// Why a snapshot was not written.
#[derive(Debug)]
pub enum SnapshotError {
    /// Python would take a path the engine does not implement (decompose); the
    /// caller must use the Python store for this write. Nothing was stored.
    Unsupported(String),
    /// The fields do not make a valid memory.
    Invalid(ModelError),
    Engine(EngineError),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Unsupported(m) => write!(f, "{m}"),
            SnapshotError::Invalid(e) => write!(f, "{e}"),
            SnapshotError::Engine(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<EngineError> for SnapshotError {
    fn from(e: EngineError) -> Self {
        SnapshotError::Engine(e)
    }
}

/// `tiers.tier_for_source`: the tier a `source` label implies when the caller
/// gives none. Unknown labels are `self-generated`.
pub fn tier_for_source(source: &str) -> &'static str {
    let s = py_strip(source).to_lowercase();
    match s.as_str() {
        "skwhisper"
        | "journal-synthesis"
        | "dreaming-engine"
        | "consolidation"
        | "claude-code-hook"
        | "mcp:memory_save_session"
        | "mcp:memory_pre_compact"
        | "observation" => "self-generated",
        "tool" | "tool-output" | "command" | "shell" | "task-pack" => "tool-output",
        "manual" | "mcp" | "cli" | "seed" | "telegram" | "notion" | "convo" | "hook" | "shared" => {
            "user-stated"
        }
        _ if s.starts_with("import") => "user-stated",
        _ if ["web", "http", "fetch", "browser", "search"]
            .iter()
            .any(|p| s.starts_with(p)) =>
        {
            "web"
        }
        _ => "self-generated",
    }
}

const AUTHORITY_TIERS: [&str; 7] = [
    "statute",
    "rule",
    "case",
    "form",
    "secondary",
    "template",
    "memory",
];

/// `retrieval.normalize_authority_tier`.
fn normalize_authority_tier(v: &str) -> &'static str {
    if v.is_empty() {
        return "memory";
    }
    let n = crate::env::casefold(py_strip(v));
    AUTHORITY_TIERS
        .iter()
        .find(|t| **t == n)
        .copied()
        .unwrap_or("memory")
}

fn py_str(v: Option<&Value>) -> String {
    match v {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(true)) => "True".into(),
        Some(Value::Bool(false)) => "False".into(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => crate::pyjson::dumps_compact(other),
    }
}

/// `retrieval.infer_authority_tier`.
pub fn infer_authority_tier(
    title: &str,
    source: &str,
    source_ref: &str,
    tags: &[String],
    metadata: &Map<String, Value>,
) -> &'static str {
    if let Some(e) = metadata.get("authority_tier") {
        let truthy = match e {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty(),
            Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        };
        if truthy {
            return normalize_authority_tier(&py_str(Some(e)));
        }
    }
    let hay = crate::env::casefold(
        &[
            title.to_string(),
            source.to_string(),
            source_ref.to_string(),
            tags.join(" "),
            py_str(metadata.get("file_path")),
            py_str(metadata.get("document_type")),
        ]
        .join(" "),
    );
    let has = |ms: &[&str]| ms.iter().any(|m| hay.contains(m));
    if has(&["usc", "u.s.c.", "cfr", "ucc", "statute", "act of"]) {
        if has(&["rule", "f.r.", "civil procedure"]) {
            return "rule";
        }
        return "statute";
    }
    if has(&[" v. ", " vs. ", "case law", "opinion", "holding"]) {
        return "case";
    }
    if has(&["form ", "fillable form", "irs form", "ao 240", "ao 239"]) {
        return "form";
    }
    if has(&["template", "generated/", "guide bundle"]) {
        return "template";
    }
    if has(&[
        "reference/",
        "american jurisprudence",
        "american jurispurdence",
        "american juris",
        "black's law",
        "handbook",
        "practitioner",
        "treatise",
    ]) {
        return "secondary";
    }
    "memory"
}

/// The valence `snapshot` infers when none was given: `0.0` with labels reads
/// them against the positive and negative sets.
pub fn infer_valence(valence: f64, labels: &[String]) -> f64 {
    if valence != 0.0 || labels.is_empty() {
        return valence;
    }
    const POSITIVE: [&str; 8] = [
        "joy",
        "trust",
        "love",
        "anticipation",
        "hope",
        "gratitude",
        "excited",
        "happy",
    ];
    const NEGATIVE: [&str; 8] = [
        "fear",
        "anger",
        "disgust",
        "sadness",
        "grief",
        "anxiety",
        "frustrated",
        "disappointed",
    ];
    let lower: std::collections::BTreeSet<String> =
        labels.iter().map(|l| crate::env::casefold(l)).collect();
    let pos = POSITIVE.iter().filter(|p| lower.contains(**p)).count() as f64;
    let neg = NEGATIVE.iter().filter(|p| lower.contains(**p)).count() as f64;
    if pos > neg {
        (0.3 + 0.2 * pos).min(1.0)
    } else if neg > pos {
        (-0.3 - 0.2 * neg).max(-1.0)
    } else {
        valence
    }
}

/// The result of [`Engine::snapshot`].
#[derive(Debug)]
pub struct Snapshot {
    pub memory: Memory,
    /// `false` when the content hash matched a memory already stored: that
    /// memory is returned untouched and nothing was written.
    pub created: bool,
}

impl Engine {
    /// `MemoryStore.snapshot` for the cases the engine implements: the dedup
    /// guard, `source_tier` from `source`, the `authority_tier` metadata key,
    /// valence inference, the seal, the flat file and the index row. `fields`
    /// are `Memory` field overrides (`layer`, `tags`, `emotional`, `source`,
    /// `source_tier`, `context_tag`, ...); `decompose` is read as Python's flag.
    ///
    /// Content of [`DECOMPOSE_MIN_LENGTH`] characters or more, or `decompose`
    /// set, is refused with [`SnapshotError::Unsupported`] before anything is
    /// stored (Python would decompose it into chunk memories).
    pub fn snapshot(
        &self,
        title: &str,
        content: &str,
        mut fields: Map<String, Value>,
        owner: &OwnerScope,
    ) -> Result<Snapshot, SnapshotError> {
        let decompose = fields
            .remove("decompose")
            .is_some_and(|v| v.as_bool().unwrap_or(false));
        // Dedup first, as Python does (before decomposition and before the write).
        if let Some(existing) = self.find_by_content(content, owner) {
            return Ok(Snapshot {
                memory: existing,
                created: false,
            });
        }
        if decompose || content.chars().count() >= DECOMPOSE_MIN_LENGTH {
            return Err(SnapshotError::Unsupported(format!(
                "content of {DECOMPOSE_MIN_LENGTH}+ characters (or decompose=True) is decomposed \
                 into chunk memories by the Python store; the Rust engine does not decompose: \
                 call the Python snapshot for this write"
            )));
        }
        let mut base = Memory::new(title, content).to_value();
        let o = base.as_object_mut().expect("memory is an object");
        let explicit_tier = fields.contains_key("source_tier");
        for (k, v) in fields {
            o.insert(k, v);
        }
        let mut m = Memory::from_stored(base).map_err(SnapshotError::Invalid)?;
        if !explicit_tier {
            m.source_tier = tier_for_source(&m.source).to_string();
        }
        let tier =
            infer_authority_tier(title, &m.source, &m.source_ref, &m.tags, &m.metadata).to_string();
        m.metadata
            .insert("authority_tier".into(), Value::String(tier));
        m.emotional.valence = infer_valence(m.emotional.valence, &m.emotional.labels);
        m.seal();
        self.save(&m, owner)?;
        Ok(Snapshot {
            memory: m,
            created: true,
        })
    }

    /// `SQLiteBackend.find_by_content_hash`: the stored memory whose content
    /// hashes the same, loaded from its flat file. The caller is the owner, so
    /// every memory is visible (Python applies the writer's reader here).
    fn find_by_content(&self, content: &str, owner: &OwnerScope) -> Option<Memory> {
        let hash = content_hash(content);
        let (id, file) = self.index().find_by_content_hash(&hash).ok().flatten()?;
        self.store()
            .load(&id, owner.reader_scope())
            .ok()
            .flatten()
            .or_else(|| self.store().read_path(std::path::Path::new(&file)).ok())
    }
}
