//! Outcome ledger: served, followed, effect per skill version (SKMEM-R-07),
//! the Rust twin of `skmemory/ledger.py`.
//!
//! * Source of truth: append-only JSONL under `<base>/ledger/`, one file per
//!   host per month (`outcomes-<host>-<YYYY-MM>.jsonl`). A host only ever
//!   appends to its own file, so Syncthing replicas never edit the same file.
//!   Readers merge every file and dedupe by the stored event id, so a replayed
//!   or duplicated line counts once. A line without a stored id gets the
//!   stable hash of its fields, the same on every host and in both engines.
//! * `index.db` gets a derived `ledger_events` table ([`Ledger::reindex`]).
//! * There is no update or delete: history is only appended to. A torn tail
//!   (a line cut off by a crash) is never fused onto: the next append starts
//!   a new line, and a torn line that does not parse is skipped on read.
//!
//! Scoring: each `followed` event with an effect is weighted by exponential
//! decay (half-life 90 days); harm counts [`HARM_WEIGHT`] times a success.
//! `follow_rate` is raw followed / served over first-party events only:
//! events whose source is in [`IMPORTED_SOURCES`] (Hermes counters) are
//! reported in `imported_*` and excluded from rates, the anti-pattern floor
//! and the served-never-followed flag. `followed_error` is a follow whose
//! matched command failed; it carries no effect (a failure is not evidence of
//! harm). A version is an anti-pattern when decayed harm exceeds decayed
//! success after at least 3 follows, and is flagged `served-never-followed`
//! when served >= 3 and followed == 0.

use crate::audience::OwnerScope;
use crate::pyjson::{float_repr, write_string};
use crate::pytime::Dt;
use crate::util::uuid4;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub mod sidecar;

pub const HALF_LIFE_DAYS: f64 = 90.0;
pub const HARM_WEIGHT: f64 = 4.0;
pub const MIN_FOLLOWS_ANTI: i64 = 3;
pub const MIN_SERVED_NEVER_FOLLOWED: i64 = 3;
pub const EVENTS: [&str; 3] = ["served", "followed", "followed_error"];
/// Counter imports from another runtime: reported apart, never counted toward floors or rates.
pub const IMPORTED_SOURCES: [&str; 1] = ["hermes-usage-counter"];
pub const EFFECTS: [&str; 3] = ["success", "harm", "neutral"];

/// Errors from the ledger. `Value` is Python's `ValueError`.
#[derive(Debug)]
pub enum LedgerError {
    Value(String),
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerError::Value(s) => write!(f, "{s}"),
            LedgerError::Io(e) => write!(f, "io: {e}"),
            LedgerError::Sqlite(e) => write!(f, "sqlite: {e}"),
        }
    }
}
impl std::error::Error for LedgerError {}
impl From<std::io::Error> for LedgerError {
    fn from(e: std::io::Error) -> Self {
        LedgerError::Io(e)
    }
}
impl From<rusqlite::Error> for LedgerError {
    fn from(e: rusqlite::Error) -> Self {
        LedgerError::Sqlite(e)
    }
}

type Result<T> = std::result::Result<T, LedgerError>;

fn verr<T>(m: impl Into<String>) -> Result<T> {
    Err(LedgerError::Value(m.into()))
}

fn warn(msg: &str) {
    eprintln!("WARNING skmem: {msg}");
}

/// The compact JSON text `json.dumps(list, separators=(",", ":"))` writes for
/// a list of strings and nulls (ASCII-escaped like Python).
fn tight_list(items: &[Option<&str>]) -> String {
    let mut o = String::from("[");
    for (n, it) in items.iter().enumerate() {
        if n > 0 {
            o.push(',');
        }
        match it {
            Some(s) => write_string(&mut o, s),
            None => o.push_str("null"),
        }
    }
    o.push(']');
    o
}

/// Stable id: the same logical event hashes the same on every host.
pub fn event_id(
    skill: &str,
    version: &str,
    event: &str,
    at: &str,
    effect: Option<&str>,
    session: &str,
) -> String {
    let raw = tight_list(&[
        Some(skill),
        Some(version),
        Some(event),
        Some(at),
        effect,
        Some(session),
    ]);
    hex32(raw.as_bytes())
}

pub(crate) fn hex32(data: &[u8]) -> String {
    let d = Sha256::digest(data);
    let h: String = d.iter().map(|b| format!("{b:02x}")).collect();
    h[..32].to_string()
}

/// Python `str(value)` of a JSON value (the ledger stringifies `version`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => py_repr(v),
    }
}

pub(crate) fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => {
            let t = n.to_string();
            if t.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
                float_repr(t.parse::<f64>().unwrap_or(f64::NAN))
            } else {
                t
            }
        }
        Value::String(s) => {
            let q = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut o = String::new();
            o.push(q);
            for c in s.chars() {
                match c {
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if c == q => {
                        o.push('\\');
                        o.push(c);
                    }
                    c => o.push(c),
                }
            }
            o.push(q);
            o
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, x)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// One deduped event as readers see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub id: String,
    pub skill: String,
    pub version: String,
    pub event: String,
    /// The `at` text as stored (not normalised).
    pub at: String,
    pub effect: Option<String>,
    pub session: String,
    pub source: Option<String>,
    pub missing: Option<String>,
}

impl Event {
    fn at_utc(&self) -> Dt {
        Dt::parse(&self.at).map_or(Dt::from_micros(0), Dt::to_utc)
    }
}

/// Score of one skill version.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub skill: String,
    pub version: String,
    pub served: i64,
    pub followed: i64,
    pub follow_rate: f64,
    pub net: f64,
    pub net_effect_sign: &'static str,
    pub anti_pattern: bool,
    pub flag: Option<&'static str>,
    pub rank: usize,
    pub imported_served: i64,
    pub imported_followed: i64,
}

impl Score {
    /// `skill@version`.
    pub fn key(&self) -> String {
        format!("{}@{}", self.skill, self.version)
    }
}

/// Optional fields of [`Ledger::record`].
#[derive(Clone, Debug, Default)]
pub struct Record<'a> {
    pub effect: Option<&'a str>,
    pub session: &'a str,
    /// Event time; default now.
    pub at: Option<Dt>,
    /// A producer that re-reads the same input supplies a stable id: a second
    /// `record` with a known id appends nothing.
    pub event_id: Option<&'a str>,
    /// What produced the event (never scored).
    pub source: Option<&'a str>,
    /// What is unknown about it (never scored). Never pass transcript text.
    pub missing: Option<&'a str>,
}

fn expanduser(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix('~') {
        if rest.is_empty() || rest.starts_with('/') {
            if let Some(h) = std::env::var_os("HOME") {
                return PathBuf::from(h).join(rest.trim_start_matches('/'));
            }
        }
    }
    p.to_path_buf()
}

fn sanitize_host(h: &str) -> String {
    h.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Append-only outcome ledger with a derived SQLite index.
#[derive(Clone, Debug)]
pub struct Ledger {
    base: PathBuf,
    dir: PathBuf,
    db_path: PathBuf,
    host: String,
}

impl Ledger {
    /// `Ledger(base_path, host)`: the host defaults to the short hostname.
    pub fn new(base: impl AsRef<Path>, host: Option<&str>, _owner: &OwnerScope) -> Ledger {
        let base = expanduser(base.as_ref());
        let raw = match host.filter(|h| !h.is_empty()) {
            Some(h) => h.to_string(),
            None => crate::sys::hostname()
                .split('.')
                .next()
                .unwrap_or("")
                .to_string(),
        };
        let h = sanitize_host(&raw);
        Ledger {
            dir: base.join("ledger"),
            db_path: base.join("index.db"),
            base,
            host: if h.is_empty() { "host".into() } else { h },
        }
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn file(&self, at: Dt) -> PathBuf {
        self.dir.join(format!(
            "outcomes-{}-{}.jsonl",
            self.host,
            at.year_month_utc()
        ))
    }

    /// Append one event to this host's file for the event's month. Returns the event id.
    pub fn record(
        &self,
        skill: &str,
        version: &str,
        event: &str,
        r: &Record<'_>,
    ) -> Result<String> {
        if !EVENTS.contains(&event) {
            return verr(format!("unknown event {event:?}"));
        }
        if let Some(e) = r.effect {
            if !EFFECTS.contains(&e) {
                return verr(format!("unknown effect {e:?}"));
            }
        }
        if event == "served" && r.effect.is_some() {
            return verr("a served event has no effect");
        }
        if skill.is_empty() || version.is_empty() {
            return verr("skill and version are required");
        }
        let when = r.at.map_or_else(Dt::now, Dt::to_utc);
        let eid = match r.event_id.filter(|e| !e.is_empty()) {
            Some(e) => e.to_string(),
            None => uuid4().replace('-', ""),
        };
        let mut line = String::from("{");
        let mut put = |k: &str, v: Option<&str>, first: bool| {
            if !first {
                line.push(',');
            }
            write_string(&mut line, k);
            line.push(':');
            match v {
                Some(s) => write_string(&mut line, s),
                None => line.push_str("null"),
            }
        };
        put("skill", Some(skill), true);
        put("version", Some(version), false);
        put("event", Some(event), false);
        put("at", Some(&when.isoformat()), false);
        put("effect", r.effect, false);
        put("session", Some(r.session), false);
        put("id", Some(&eid), false);
        let source: Option<String> = r
            .source
            .filter(|s| !s.is_empty())
            .map(|s| s.chars().take(80).collect());
        let missing: Option<String> = r
            .missing
            .filter(|s| !s.is_empty())
            .map(|s| s.chars().take(200).collect());
        if let Some(s) = &source {
            put("source", Some(s), false);
        }
        if let Some(m) = &missing {
            put("missing", Some(m), false);
        }
        line.push_str("}\n");
        if r.event_id.is_some_and(|e| !e.is_empty()) && self.known(&eid) {
            return Ok(eid);
        }
        fs::create_dir_all(&self.dir)?;
        let path = self.file(when);
        let mut f = OpenOptions::new().append(true).create(true).open(&path)?;
        let mut buf = Vec::new();
        if f.metadata()?.len() > 0 {
            let mut rd = fs::File::open(&path)?;
            rd.seek(SeekFrom::End(-1))?;
            let mut last = [0u8; 1];
            rd.read_exact(&mut last)?;
            if last[0] != b'\n' {
                buf.push(b'\n'); // torn tail: do not fuse onto it
            }
        }
        buf.extend_from_slice(line.as_bytes());
        f.write_all(&buf)?;
        drop(f);
        let rec = Event {
            id: eid.clone(),
            skill: skill.into(),
            version: version.into(),
            event: event.into(),
            at: when.isoformat(),
            effect: r.effect.map(str::to_string),
            session: r.session.into(),
            source,
            missing,
        };
        self.index(&[rec]);
        Ok(eid)
    }

    /// True when the derived index already holds this id (best effort: a miss
    /// just appends a line that readers dedupe by id).
    pub fn known(&self, eid: &str) -> bool {
        if !self.db_path.is_file() {
            return false;
        }
        let run = || -> rusqlite::Result<bool> {
            let c = self.conn()?;
            Ok(
                c.query_row("SELECT 1 FROM ledger_events WHERE id=?", [eid], |_| Ok(()))
                    .optional()?
                    .is_some(),
            )
        };
        run().unwrap_or(false)
    }

    /// Every event from every host file, deduped by id, ordered by time.
    pub fn events(&self) -> Vec<Event> {
        let mut files: Vec<PathBuf> = match fs::read_dir(&self.dir) {
            Ok(rd) => rd
                .filter_map(std::result::Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    let n = p.file_name().map(|n| n.to_string_lossy().into_owned());
                    n.is_some_and(|n| n.starts_with("outcomes-") && n.ends_with(".jsonl"))
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        files.sort();
        let mut seen: HashMap<String, Event> = HashMap::new();
        for p in files {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let Ok(bytes) = fs::read(&p) else { continue };
            for (n, raw) in bytes.split(|b| *b == b'\n').enumerate() {
                if raw.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                match parse_line(raw) {
                    Ok(ev) => {
                        seen.entry(ev.id.clone()).or_insert(ev);
                    }
                    Err(err) => warn(&format!("ledger: skipping {name}:{} ({err})", n + 1)),
                }
            }
        }
        let mut v: Vec<Event> = seen.into_values().collect();
        v.sort_by(|a, b| (a.at_utc(), &a.id).cmp(&(b.at_utc(), &b.id)));
        v
    }

    fn conn(&self) -> rusqlite::Result<Connection> {
        let _ = fs::create_dir_all(&self.base);
        let c = Connection::open(&self.db_path)?;
        c.busy_timeout(Duration::from_secs(30))?;
        c.execute_batch(
            "CREATE TABLE IF NOT EXISTS ledger_events (id TEXT PRIMARY KEY, skill TEXT, \
             version TEXT, event TEXT, at TEXT, effect TEXT, session TEXT)",
        )?;
        Ok(c)
    }

    /// The index is derived: a failure here must never lose the appended event.
    fn index(&self, recs: &[Event]) {
        let run = || -> rusqlite::Result<()> {
            let mut c = self.conn()?;
            let tx = c.transaction()?;
            for r in recs {
                tx.execute(
                    "INSERT OR IGNORE INTO ledger_events VALUES (?,?,?,?,?,?,?)",
                    params![r.id, r.skill, r.version, r.event, r.at, r.effect, r.session],
                )?;
            }
            tx.commit()
        };
        if let Err(e) = run() {
            warn(&format!("ledger: index update failed ({e}); run reindex"));
        }
    }

    /// Rebuild `ledger_events` from the JSONL files. Call after a Syncthing pull.
    pub fn reindex(&self) -> Result<usize> {
        let recs = self.events();
        let mut c = self.conn()?;
        let tx = c.transaction()?;
        tx.execute("DELETE FROM ledger_events", [])?;
        for r in &recs {
            tx.execute(
                "INSERT INTO ledger_events VALUES (?,?,?,?,?,?,?)",
                params![r.id, r.skill, r.version, r.event, r.at, r.effect, r.session],
            )?;
        }
        tx.commit()?;
        Ok(recs.len())
    }

    /// Score every skill version, ranked by decayed net effect (best first).
    pub fn scores(&self, as_of: Option<Dt>) -> Vec<Score> {
        score_events(&self.events(), as_of.map_or_else(Dt::now, Dt::to_utc))
    }

    pub fn score(&self, skill: &str, version: &str, as_of: Option<Dt>) -> Option<Score> {
        self.scores(as_of)
            .into_iter()
            .find(|s| s.skill == skill && s.version == version)
    }

    pub fn top(&self, as_of: Option<Dt>, limit: usize) -> Vec<Score> {
        let mut s = self.scores(as_of);
        s.truncate(limit);
        s
    }
}

fn parse_line(raw: &[u8]) -> std::result::Result<Event, String> {
    let text = std::str::from_utf8(raw).map_err(|e| e.to_string())?;
    let r: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let o = r.as_object().ok_or("not an object")?;
    let get = |k: &str| o.get(k).ok_or_else(|| format!("missing {k}"));
    let skill = get("skill")?;
    let version = py_str(get("version")?);
    let event = get("event")?;
    let at = get("at")?;
    get("session")?;
    let (Some(skill), Some(event), Some(at), Some(session)) = (
        skill.as_str(),
        event.as_str(),
        at.as_str(),
        o.get("session").and_then(Value::as_str),
    ) else {
        // The Python takes non-string skill/session as they come; a ledger
        // line is only ever written with strings, so a stranger is skipped.
        return Err("non-string field".into());
    };
    if !EVENTS.contains(&event) {
        return Err("bad event".into());
    }
    if Dt::parse(at).is_none() {
        return Err("bad at".into());
    }
    let effect = o.get("effect").and_then(Value::as_str);
    let id = match o.get("id").and_then(Value::as_str) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => event_id(skill, &version, event, at, effect, session),
    };
    let extra = |k: &str| {
        o.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    Ok(Event {
        id,
        skill: skill.into(),
        version,
        event: event.into(),
        at: at.into(),
        effect: effect.map(str::to_string),
        session: session.into(),
        source: extra("source"),
        missing: extra("missing"),
    })
}

/// Score a set of events as of `now` (see the module docs for the rules).
pub fn score_events(events: &[Event], now: Dt) -> Vec<Score> {
    // [served, followed, decayed success, decayed harm, imported served, imported followed]
    let mut agg: Vec<((String, String), [f64; 6])> = Vec::new();
    let mut idx: HashMap<(String, String), usize> = HashMap::new();
    for r in events {
        let key = (r.skill.clone(), r.version.clone());
        let n = *idx.entry(key.clone()).or_insert_with(|| {
            agg.push((key, [0.0; 6]));
            agg.len() - 1
        });
        let a = &mut agg[n].1;
        if r.source
            .as_deref()
            .is_some_and(|s| IMPORTED_SOURCES.contains(&s))
        {
            a[if r.event == "served" { 4 } else { 5 }] += 1.0;
            continue;
        }
        if r.event == "served" {
            a[0] += 1.0;
            continue;
        }
        a[1] += 1.0; // followed and followed_error both count as a (heuristic) follow
                     // Future-dated events (clock skew between hosts) are capped at weight 1.0.
        let age = (((now.micros - r.at_utc().micros) as f64) / 1e6 / 86400.0).max(0.0);
        let w = 0.5f64.powf(age / HALF_LIFE_DAYS);
        match r.effect.as_deref() {
            Some("success") => a[2] += w,
            Some("harm") => a[3] += w,
            _ => {}
        }
    }
    let mut out: Vec<Score> = agg
        .into_iter()
        .map(|((skill, version), a)| {
            let (served, followed, succ, harm) = (a[0] as i64, a[1] as i64, a[2], a[3]);
            let net = succ - HARM_WEIGHT * harm;
            Score {
                skill,
                version,
                served,
                followed,
                imported_served: a[4] as i64,
                imported_followed: a[5] as i64,
                follow_rate: if served != 0 {
                    followed as f64 / served as f64
                } else {
                    0.0
                },
                net,
                net_effect_sign: if net > 1e-9 {
                    "+"
                } else if net < -1e-9 {
                    "-"
                } else {
                    "0"
                },
                anti_pattern: followed >= MIN_FOLLOWS_ANTI && harm > succ,
                flag: (served >= MIN_SERVED_NEVER_FOLLOWED && followed == 0)
                    .then_some("served-never-followed"),
                rank: 0,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        (-a.net)
            .partial_cmp(&(-b.net))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.key().cmp(&b.key()))
    });
    for (i, s) in out.iter_mut().enumerate() {
        s.rank = i + 1;
    }
    out
}
