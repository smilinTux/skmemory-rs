//! The skmemory `Memory` record, schema v2, byte-compatible with Python.
//!
//! The Python source of truth is `skmemory/models.py` (pydantic). This module
//! reproduces what Python does on a read from storage (`Memory.from_stored`)
//! and on a dump (`json.dumps(memory.model_dump(), indent=2)`):
//!
//! * declared fields are written in declaration order, then unknown fields
//!   (`extra="allow"`) in the order they were read, so a node that loads and
//!   re-saves a newer file does not strip them;
//! * a `kind`, `outcome` or `source_tier` outside the v2 vocabulary loads as
//!   the default with the raw value kept in `metadata.schema_v2_unknown`, and
//!   is written back verbatim on save while the field still holds the default;
//! * a file with no `schema_version` is a v1 file and reports 1 in memory;
//!   nothing is rewritten on read;
//! * `event_at` and `valid_from` default to `created_at` in memory;
//! * nested `emotional` ignores unknown keys (pydantic default), as Python does.
//!
//! The model is a hand-written codec over `serde_json::Value` rather than a
//! `#[derive(Serialize)]`: byte parity needs Python's float and string
//! formatting and the lenient rules above, which a derive cannot express.

use crate::pyjson;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;

/// Current on-disk schema version written for new and re-saved memories.
pub const SCHEMA_VERSION: i64 = 2;

/// Key in `metadata` that holds raw out-of-vocabulary v2 values.
pub const UNKNOWN_KEY: &str = "schema_v2_unknown";

const KINDS: &[&str] = &["episode", "fact", "procedure", "preference", "rule"];
const OUTCOMES: &[&str] = &["success", "failure", "lesson"];
const TIERS: &[&str] = &["user-stated", "tool-output", "web", "self-generated"];

/// Persistence tier; the directory name under `<agent>/memory/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    Short,
    Mid,
    Long,
}

impl Layer {
    /// All layers in Python's iteration order.
    pub const ALL: [Layer; 3] = [Layer::Short, Layer::Mid, Layer::Long];

    /// The on-disk value and directory name.
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::Short => "short-term",
            Layer::Mid => "mid-term",
            Layer::Long => "long-term",
        }
    }

    /// Inverse of [`Layer::as_str`].
    pub fn parse(s: &str) -> Option<Layer> {
        Layer::ALL.into_iter().find(|l| l.as_str() == s)
    }
}

/// Role-based partitioning (`MemoryRole`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Dev,
    Ops,
    Sec,
    Ai,
    General,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Dev => "dev",
            Role::Ops => "ops",
            Role::Sec => "sec",
            Role::Ai => "ai",
            Role::General => "general",
        }
    }

    fn parse(s: &str) -> Option<Role> {
        [Role::Dev, Role::Ops, Role::Sec, Role::Ai, Role::General]
            .into_iter()
            .find(|r| r.as_str() == s)
    }
}

/// The feeling component of a memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Emotional {
    pub intensity: f64,
    pub valence: f64,
    pub labels: Vec<String>,
    pub resonance_note: String,
    pub cloud9_achieved: bool,
}

impl Default for Emotional {
    fn default() -> Self {
        Emotional {
            intensity: 0.0,
            valence: 0.0,
            labels: Vec::new(),
            resonance_note: String::new(),
            cloud9_achieved: false,
        }
    }
}

impl Emotional {
    /// Python `EmotionalSnapshot.signature()`, e.g. `joy+trust@8.5/+0.9`.
    pub fn signature(&self) -> String {
        let labels = if self.labels.is_empty() {
            "neutral".to_string()
        } else {
            let mut l: Vec<&str> = self.labels.iter().map(String::as_str).collect();
            l.sort_unstable();
            l.join("+")
        };
        format!("{}@{:.1}/{:+.1}", labels, self.intensity, self.valence)
    }
}

/// A single memory. Field order here is the on-disk key order.
#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub layer: Layer,
    pub role: Role,
    pub title: String,
    pub content: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub source: String,
    pub source_ref: String,
    pub emotional: Emotional,
    pub related_ids: Vec<String>,
    pub parent_id: Option<String>,
    pub context_tag: String,
    pub intent: String,
    pub integrity_hash: String,
    pub metadata: Map<String, Value>,
    pub kind: String,
    pub outcome: Option<String>,
    pub event_at: Option<String>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub supersedes_id: Option<String>,
    pub source_tier: String,
    pub schema_version: i64,
    /// Fields this version does not know about, in read order.
    pub extra: Map<String, Value>,
}

/// Why a stored record could not become a `Memory` (Python would raise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError(msg.into()))
}

/// Python `str.strip()`: Unicode whitespace plus the ASCII separators 0x1c-0x1f.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn take_str(data: &mut Map<String, Value>, key: &str) -> Result<Option<String>, ModelError> {
    match data.shift_remove(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => err(format!("{key}: expected a string")),
    }
}

fn take_opt_str(
    data: &mut Map<String, Value>,
    key: &str,
) -> Result<Option<Option<String>>, ModelError> {
    match data.shift_remove(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(s)) => Ok(Some(Some(s))),
        Some(_) => err(format!("{key}: expected a string or null")),
    }
}

fn take_str_list(
    data: &mut Map<String, Value>,
    key: &str,
) -> Result<Option<Vec<String>>, ModelError> {
    match data.shift_remove(key) {
        None => Ok(None),
        Some(Value::Array(items)) => items
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(s),
                _ => err(format!("{key}: expected a list of strings")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => err(format!("{key}: expected a list")),
    }
}

fn float_in(v: &Value, key: &str, lo: f64, hi: f64) -> Result<f64, ModelError> {
    match v {
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            if x >= lo && x <= hi {
                Ok(x)
            } else {
                err(format!("{key}: out of range"))
            }
        }
        _ => err(format!("{key}: expected a number")),
    }
}

fn parse_emotional(v: Value) -> Result<Emotional, ModelError> {
    let Value::Object(mut m) = v else {
        return err("emotional: expected an object");
    };
    let mut e = Emotional::default();
    if let Some(x) = m.shift_remove("intensity") {
        e.intensity = float_in(&x, "emotional.intensity", 0.0, 10.0)?;
    }
    if let Some(x) = m.shift_remove("valence") {
        e.valence = float_in(&x, "emotional.valence", -1.0, 1.0)?;
    }
    if let Some(l) = take_str_list(&mut m, "labels")? {
        e.labels = l;
    }
    if let Some(s) = take_str(&mut m, "resonance_note")? {
        e.resonance_note = s;
    }
    if let Some(b) = m.shift_remove("cloud9_achieved") {
        e.cloud9_achieved = match b {
            Value::Bool(b) => b,
            Value::Number(n) if n.as_i64() == Some(0) => false,
            Value::Number(n) if n.as_i64() == Some(1) => true,
            _ => return err("emotional.cloud9_achieved: expected a bool"),
        };
    }
    // Unknown keys inside `emotional` are ignored, as pydantic's default does.
    Ok(e)
}

/// Apply the read-from-storage leniency to the raw object (Python's
/// `_tolerate_stored_values`).
fn apply_lenient(data: &mut Map<String, Value>) -> Result<(), ModelError> {
    if !data.contains_key("schema_version") {
        data.insert("schema_version".into(), Value::from(1));
    }
    let vocab: [(&str, Value, &[&str], bool); 3] = [
        ("kind", Value::from("episode"), KINDS, false),
        ("outcome", Value::Null, OUTCOMES, true),
        ("source_tier", Value::from("self-generated"), TIERS, false),
    ];
    for (field, default, allowed, nullable) in vocab {
        let Some(raw) = data.get(field).cloned() else {
            continue;
        };
        let ok = match &raw {
            Value::String(s) => allowed.contains(&s.as_str()),
            Value::Null => nullable,
            _ => false,
        };
        if ok {
            continue;
        }
        let mut meta = match data.get("metadata") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return err("metadata: expected an object"),
        };
        let mut unknown = match meta.get(UNKNOWN_KEY) {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return err("metadata.schema_v2_unknown: expected an object"),
        };
        unknown.insert(field.into(), raw);
        meta.insert(UNKNOWN_KEY.into(), Value::Object(unknown));
        data.insert("metadata".into(), Value::Object(meta));
        data.insert(field.into(), default);
    }
    Ok(())
}

impl Memory {
    /// A new memory with Python's field defaults (not yet sealed or saved).
    pub fn new(title: &str, content: &str) -> Memory {
        let now = crate::util::now_iso();
        Memory {
            id: crate::util::uuid4(),
            created_at: now.clone(),
            updated_at: now.clone(),
            layer: Layer::Short,
            role: Role::General,
            title: py_strip(title).to_string(),
            content: content.to_string(),
            summary: String::new(),
            tags: Vec::new(),
            source: "manual".into(),
            source_ref: String::new(),
            emotional: Emotional::default(),
            related_ids: Vec::new(),
            parent_id: None,
            context_tag: "@chef-only".into(),
            intent: String::new(),
            integrity_hash: String::new(),
            metadata: Map::new(),
            kind: "episode".into(),
            outcome: None,
            event_at: Some(now.clone()),
            valid_from: Some(now),
            valid_to: None,
            supersedes_id: None,
            source_tier: "self-generated".into(),
            schema_version: SCHEMA_VERSION,
            extra: Map::new(),
        }
    }

    /// Python `Memory.from_stored_json`: parse bytes read from storage.
    pub fn from_stored_json(raw: &[u8]) -> Result<Memory, ModelError> {
        let text = std::str::from_utf8(raw).map_err(|e| ModelError(format!("utf-8: {e}")))?;
        let v: Value = serde_json::from_str(text)
            .map_err(|e| ModelError(format!("json: {e}{}", unsupported_hint(text))))?;
        Memory::from_stored(v)
    }

    /// Python `Memory.from_stored`: build from a stored JSON object, applying
    /// the lenient v2 rules. Errors where Python's validation would raise.
    pub fn from_stored(value: Value) -> Result<Memory, ModelError> {
        let Value::Object(mut data) = value else {
            return err("record: expected a JSON object");
        };
        apply_lenient(&mut data)?;
        let d = &mut data;

        let id = match take_str(d, "id")? {
            Some(s) => s,
            None => crate::util::uuid4(),
        };
        let id = py_strip(&id).to_string();
        if id.is_empty() {
            return err("Memory ID cannot be empty or null");
        }
        let now = crate::util::now_iso();
        let created_at = take_str(d, "created_at")?.unwrap_or_else(|| now.clone());
        let updated_at = take_str(d, "updated_at")?.unwrap_or(now);
        let layer = match take_str(d, "layer")? {
            None => Layer::Short,
            Some(s) => Layer::parse(&s).ok_or_else(|| ModelError(format!("layer: {s:?}")))?,
        };
        let role = match take_str(d, "role")? {
            None => Role::General,
            Some(s) => Role::parse(&s).ok_or_else(|| ModelError(format!("role: {s:?}")))?,
        };
        let Some(title) = take_str(d, "title")? else {
            return err("title: required");
        };
        let title = py_strip(&title).to_string();
        if title.is_empty() {
            return err("Memory title cannot be empty");
        }
        let Some(content) = take_str(d, "content")? else {
            return err("content: required");
        };
        let summary = take_str(d, "summary")?.unwrap_or_default();
        let tags = take_str_list(d, "tags")?.unwrap_or_default();
        let source = take_str(d, "source")?.unwrap_or_else(|| "manual".into());
        let source_ref = take_str(d, "source_ref")?.unwrap_or_default();
        let emotional = match d.shift_remove("emotional") {
            None => Emotional::default(),
            Some(v) => parse_emotional(v)?,
        };
        let related_ids = take_str_list(d, "related_ids")?.unwrap_or_default();
        let parent_id = take_opt_str(d, "parent_id")?.unwrap_or(None);
        let context_tag = take_str(d, "context_tag")?.unwrap_or_else(|| "@chef-only".into());
        let intent = take_str(d, "intent")?.unwrap_or_default();
        let integrity_hash = take_str(d, "integrity_hash")?.unwrap_or_default();
        let metadata = match d.shift_remove("metadata") {
            None => Map::new(),
            Some(Value::Object(m)) => m,
            Some(_) => return err("metadata: expected an object"),
        };
        let kind = take_str(d, "kind")?.unwrap_or_else(|| "episode".into());
        let outcome = take_opt_str(d, "outcome")?.unwrap_or(None);
        let event_at = take_opt_str(d, "event_at")?.unwrap_or(None);
        let valid_from = take_opt_str(d, "valid_from")?.unwrap_or(None);
        let valid_to = take_opt_str(d, "valid_to")?.unwrap_or(None);
        let supersedes_id = take_opt_str(d, "supersedes_id")?.unwrap_or(None);
        let source_tier = take_str(d, "source_tier")?.unwrap_or_else(|| "self-generated".into());
        let schema_version = match d.shift_remove("schema_version") {
            None => SCHEMA_VERSION,
            Some(Value::Number(n)) => match (n.as_i64(), n.as_f64()) {
                (Some(i), _) => i,
                (None, Some(f)) if f.fract() == 0.0 && f.abs() < 9e15 => f as i64,
                _ => return err("schema_version: expected an integer"),
            },
            Some(_) => return err("schema_version: expected an integer"),
        };
        let event_at = event_at.or_else(|| Some(created_at.clone()));
        let valid_from = valid_from.or_else(|| Some(created_at.clone()));
        Ok(Memory {
            id,
            created_at,
            updated_at,
            layer,
            role,
            title,
            content,
            summary,
            tags,
            source,
            source_ref,
            emotional,
            related_ids,
            parent_id,
            context_tag,
            intent,
            integrity_hash,
            metadata,
            kind,
            outcome,
            event_at,
            valid_from,
            valid_to,
            supersedes_id,
            source_tier,
            schema_version,
            extra: data,
        })
    }

    /// Python `compute_integrity_hash`: SHA-256 over
    /// `id:title:content:emotional-signature`, 64 lowercase hex chars.
    pub fn compute_integrity_hash(&self) -> String {
        let payload = format!(
            "{}:{}:{}:{}",
            self.id,
            self.title,
            self.content,
            self.emotional.signature()
        );
        let digest = Sha256::digest(payload.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Compute and store the integrity hash.
    pub fn seal(&mut self) {
        self.integrity_hash = self.compute_integrity_hash();
    }

    /// An unsealed memory verifies; a sealed one must match.
    pub fn verify_integrity(&self) -> bool {
        self.integrity_hash.is_empty() || self.integrity_hash == self.compute_integrity_hash()
    }

    /// The memory as an ordered JSON object, i.e. Python's `model_dump()`
    /// including the wrap serializer that restores raw unknown v2 values.
    pub fn to_value(&self) -> Value {
        let mut o = Map::new();
        let s = |x: &str| Value::String(x.to_string());
        let os = |x: &Option<String>| x.as_ref().map_or(Value::Null, |v| s(v));
        let list = |x: &[String]| Value::Array(x.iter().map(|v| s(v)).collect());
        let fl = |x: f64| Value::Number(serde_json::Number::from_f64(x).unwrap_or(0.into()));
        o.insert("id".into(), s(&self.id));
        o.insert("created_at".into(), s(&self.created_at));
        o.insert("updated_at".into(), s(&self.updated_at));
        o.insert("layer".into(), s(self.layer.as_str()));
        o.insert("role".into(), s(self.role.as_str()));
        o.insert("title".into(), s(&self.title));
        o.insert("content".into(), s(&self.content));
        o.insert("summary".into(), s(&self.summary));
        o.insert("tags".into(), list(&self.tags));
        o.insert("source".into(), s(&self.source));
        o.insert("source_ref".into(), s(&self.source_ref));
        let mut e = Map::new();
        e.insert("intensity".into(), fl(self.emotional.intensity));
        e.insert("valence".into(), fl(self.emotional.valence));
        e.insert("labels".into(), list(&self.emotional.labels));
        e.insert("resonance_note".into(), s(&self.emotional.resonance_note));
        e.insert(
            "cloud9_achieved".into(),
            Value::Bool(self.emotional.cloud9_achieved),
        );
        o.insert("emotional".into(), Value::Object(e));
        o.insert("related_ids".into(), list(&self.related_ids));
        o.insert("parent_id".into(), os(&self.parent_id));
        o.insert("context_tag".into(), s(&self.context_tag));
        o.insert("intent".into(), s(&self.intent));
        o.insert("integrity_hash".into(), s(&self.integrity_hash));
        o.insert("metadata".into(), Value::Object(self.metadata.clone()));
        o.insert("kind".into(), s(&self.kind));
        o.insert("outcome".into(), os(&self.outcome));
        o.insert("event_at".into(), os(&self.event_at));
        o.insert("valid_from".into(), os(&self.valid_from));
        o.insert("valid_to".into(), os(&self.valid_to));
        o.insert("supersedes_id".into(), os(&self.supersedes_id));
        o.insert("source_tier".into(), s(&self.source_tier));
        o.insert("schema_version".into(), Value::from(self.schema_version));
        for (k, v) in &self.extra {
            o.insert(k.clone(), v.clone());
        }
        restore_unknown(&mut o);
        Value::Object(o)
    }

    /// Exactly what Python's `FileBackend.save` writes:
    /// `json.dumps(memory.model_dump(), indent=2, default=str)`.
    pub fn to_json(&self) -> String {
        pyjson::dumps_indent2(&self.to_value())
    }
}

/// Python's `_restore_unknown_on_dump`: put raw out-of-vocabulary values back
/// while the field still holds its default, then drop emptied bookkeeping.
fn restore_unknown(out: &mut Map<String, Value>) {
    let Some(Value::Object(meta)) = out.get("metadata") else {
        return;
    };
    let mut meta = meta.clone();
    let unknown = match meta.get(UNKNOWN_KEY) {
        Some(Value::Object(u)) if !u.is_empty() => u.clone(),
        _ => return,
    };
    let mut remaining = unknown.clone();
    for (field, raw) in &unknown {
        let default = match field.as_str() {
            "kind" => Value::from("episode"),
            "outcome" => Value::Null,
            "source_tier" => Value::from("self-generated"),
            _ => continue,
        };
        if out.get(field) == Some(&default) {
            out.insert(field.clone(), raw.clone());
            remaining.shift_remove(field);
        }
    }
    if remaining.is_empty() {
        meta.shift_remove(UNKNOWN_KEY);
    } else {
        meta.insert(UNKNOWN_KEY.into(), Value::Object(remaining));
    }
    out.insert("metadata".into(), Value::Object(meta));
}

/// When parsing fails, say whether the cause is a construct Python accepts but
/// this reader cannot represent: bare `NaN`/`Infinity` tokens, or a lone
/// UTF-16 surrogate escape inside a string. Returns "" when neither is found.
fn unsupported_hint(text: &str) -> &'static str {
    let b = text.as_bytes();
    let mut i = 0;
    let (mut nonfinite, mut surrogate) = (false, false);
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        if b.get(i + 1) == Some(&b'u') && i + 6 <= b.len() {
                            if let Ok(u) = text
                                .get(i + 2..i + 6)
                                .map_or(Err(()), |h| u16::from_str_radix(h, 16).map_err(|_| ()))
                            {
                                let lone_high = (0xD800..0xDC00).contains(&u)
                                    && !(b.get(i + 6) == Some(&b'\\')
                                        && b.get(i + 7) == Some(&b'u')
                                        && text
                                            .get(i + 8..i + 12)
                                            .and_then(|h| u16::from_str_radix(h, 16).ok())
                                            .is_some_and(|l| (0xDC00..0xE000).contains(&l)));
                                let lone_low = (0xDC00..0xE000).contains(&u)
                                    && !(i >= 6
                                        && b[i - 6] == b'\\'
                                        && text
                                            .get(i - 4..i)
                                            .and_then(|h| u16::from_str_radix(h, 16).ok())
                                            .is_some_and(|h| (0xD800..0xDC00).contains(&h)));
                                surrogate |= lone_high || lone_low;
                            }
                            i += 6;
                        } else {
                            i += 2;
                        }
                    } else {
                        i += 1;
                    }
                }
                i += 1;
            }
            b'N' if text.as_bytes()[i..].starts_with(b"NaN") => {
                nonfinite = true;
                i += 3;
            }
            b'I' if text.as_bytes()[i..].starts_with(b"Infinity") => {
                nonfinite = true;
                i += 8;
            }
            _ => i += 1,
        }
    }
    match (nonfinite, surrogate) {
        (true, _) => " (unsupported: NaN/Infinity literal)",
        (_, true) => " (unsupported: lone surrogate escape)",
        _ => "",
    }
}
