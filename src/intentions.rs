//! Intentions: typed prospective memory with a code-owned lifecycle
//! (SKMEM-R-07), the Rust twin of `skmemory/intentions.py`.
//!
//! Python and Rust engines share the same files during a shadow run, so every
//! on-disk and cross-process contract is matched exactly:
//!
//! * the flat file `<base>/intentions/<id>.json` is the source of truth and
//!   is the text pydantic's `model_dump_json(indent=2)` writes (field order,
//!   `Z` for UTC, extra fields kept after the declared ones);
//! * the derived `intentions` table in `<base>/index.db` has the same DDL,
//!   the same columns and the same cell values;
//! * the per-intention `flock` lives in a host-local directory,
//!   `<XDG_RUNTIME_DIR or tempdir>/skmemory-intent-locks-<uid>/<sha1(resolved base)[:16]>/<id>.lock`,
//!   so a Python and a Rust process contend on one lock file. The tick lock
//!   is `_tick.lock` in the same directory. If these paths ever differed the
//!   two engines would race, so [`IntentionStore::lock_dir`] is the contract
//!   and is tested against Python;
//! * the state machine, merge lattice, prefilter and supersession sweep follow
//!   the Python line for line (see the per-item notes).

use crate::audience::OwnerScope;
use crate::flat::atomic_write_bytes;
use crate::model::py_strip;
use crate::pytime::{Dt, EPOCH};
use crate::sys;
use crate::util::is_py_word_char;
use rusqlite::{params_from_iter, types::Value as SqlValue, Connection};
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

pub mod tick;

/// Terminal states ignore every event.
pub const TERMINAL_STATES: [&str; 4] = ["done", "cancelled", "expired", "invalidated"];
const LAYERS: [&str; 3] = ["short-term", "mid-term", "long-term"];
const SEP: char = '\x1f';
/// Default prefilter deadline, seconds (`SKMEMORY_PREFILTER_DEADLINE_S`).
pub const DEFAULT_DEADLINE_S: f64 = 0.05;
/// Default fire cap per prompt (`SKMEMORY_PREFILTER_MAX_FIRES`).
pub const DEFAULT_MAX_FIRES: i64 = 3;
/// Default scanned head+tail size in chars (`SKMEMORY_PREFILTER_MAX_CHARS`).
pub const DEFAULT_MAX_CHARS: i64 = 16384;

/// Errors from the store. `Value` is Python's `ValueError`.
#[derive(Debug)]
pub enum IntentionError {
    Value(String),
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
}

impl fmt::Display for IntentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntentionError::Value(s) => write!(f, "{s}"),
            IntentionError::Io(e) => write!(f, "io: {e}"),
            IntentionError::Sqlite(e) => write!(f, "sqlite: {e}"),
        }
    }
}
impl std::error::Error for IntentionError {}
impl From<std::io::Error> for IntentionError {
    fn from(e: std::io::Error) -> Self {
        IntentionError::Io(e)
    }
}
impl From<rusqlite::Error> for IntentionError {
    fn from(e: rusqlite::Error) -> Self {
        IntentionError::Sqlite(e)
    }
}

pub(crate) type Result<T> = std::result::Result<T, IntentionError>;

/// Python `repr(str)`.
pub(crate) fn rp(s: &str) -> String {
    crate::ledger::py_repr(&Value::String(s.into()))
}

fn verr<T>(msg: impl Into<String>) -> Result<T> {
    Err(IntentionError::Value(msg.into()))
}

pub(crate) fn warn(msg: &str) {
    eprintln!("WARNING skmem: {msg}");
}

// ---------------------------------------------------------------------------
// pydantic-style JSON writer
// ---------------------------------------------------------------------------

/// The JSON text pydantic's `model_dump_json` writes: UTF-8 kept as is (no
/// `ensure_ascii`), only `"`, `\`, and control characters escaped, `,` and `:`
/// separators when compact, `indent=2` otherwise, floats as `ryu` prints them.
pub(crate) fn pyd_dump(v: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    pyd_write(&mut out, v, indent, 0);
    out
}

fn pyd_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn pyd_num(n: &serde_json::Number) -> String {
    let t = n.to_string();
    if t.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        let f = t.parse::<f64>().unwrap_or(f64::NAN);
        format!("{f:?}")
    } else {
        t
    }
}

fn pyd_nl(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    for _ in 0..indent * level {
        out.push(' ');
    }
}

fn pyd_write(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&pyd_num(n)),
        Value::String(s) => pyd_str(out, s),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Some(n) = indent {
                    pyd_nl(out, n, level + 1);
                }
                pyd_write(out, x, indent, level + 1);
            }
            if let Some(n) = indent {
                pyd_nl(out, n, level);
            }
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Some(n) = indent {
                    pyd_nl(out, n, level + 1);
                }
                pyd_str(out, k);
                out.push(':');
                if indent.is_some() {
                    out.push(' ');
                }
                pyd_write(out, x, indent, level + 1);
            }
            if let Some(n) = indent {
                pyd_nl(out, n, level);
            }
            out.push('}');
        }
    }
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

/// `time` (due_at) or `event` (trigger_terms) intention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Time,
    Event,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Time => "time",
            Kind::Event => "event",
        }
    }
}

/// Lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Pending,
    Armed,
    Fired,
    Done,
    Cancelled,
    Expired,
    Invalidated,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Pending => "pending",
            State::Armed => "armed",
            State::Fired => "fired",
            State::Done => "done",
            State::Cancelled => "cancelled",
            State::Expired => "expired",
            State::Invalidated => "invalidated",
        }
    }

    pub fn parse(s: &str) -> Option<State> {
        Some(match s {
            "pending" => State::Pending,
            "armed" => State::Armed,
            "fired" => State::Fired,
            "done" => State::Done,
            "cancelled" => State::Cancelled,
            "expired" => State::Expired,
            "invalidated" => State::Invalidated,
            _ => return None,
        })
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            State::Done | State::Cancelled | State::Expired | State::Invalidated
        )
    }

    /// Merge precedence: terminal beats live, `done > cancelled > invalidated >
    /// expired`, `fired > armed > pending`.
    fn rank(self) -> u8 {
        match self {
            State::Pending => 0,
            State::Armed => 1,
            State::Fired => 2,
            State::Expired => 10,
            State::Invalidated => 11,
            State::Cancelled => 12,
            State::Done => 13,
        }
    }
}

/// A prospective memory. Unknown fields are kept (`extra`) so a newer host's
/// data survives a rewrite.
#[derive(Clone, Debug, PartialEq)]
pub struct Intention {
    pub id: String,
    pub kind: Kind,
    pub title: String,
    pub state: State,
    pub created_at: Dt,
    pub origin_session: Option<String>,
    pub origin_memory_id: Option<String>,
    pub channel: String,
    pub due_at: Option<Dt>,
    pub snooze_until: Option<Dt>,
    /// Repeat every this many seconds: a fire re-arms with `due_at` advanced.
    pub interval_s: Option<i64>,
    pub trigger_terms: Vec<String>,
    /// Insertion-ordered `dict[str, str]`.
    pub scope: Vec<(String, String)>,
    pub cooldown_s: i64,
    pub max_fires: Option<i64>,
    pub fires: i64,
    pub last_fired_at: Option<Dt>,
    pub expires_at: Option<Dt>,
    /// Fields this model does not declare, in file order.
    pub extra: Map<String, Value>,
}

const DECLARED: [&str; 18] = [
    "id",
    "kind",
    "title",
    "state",
    "created_at",
    "origin_session",
    "origin_memory_id",
    "channel",
    "due_at",
    "snooze_until",
    "interval_s",
    "trigger_terms",
    "scope",
    "cooldown_s",
    "max_fires",
    "fires",
    "last_fired_at",
    "expires_at",
];

fn want_str(m: &Map<String, Value>, k: &str) -> std::result::Result<Option<String>, String> {
    match m.get(k) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("{k}: expected a string")),
    }
}

fn want_opt_str(m: &Map<String, Value>, k: &str) -> std::result::Result<Option<String>, String> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("{k}: expected a string or null")),
    }
}

fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| {
            let f = n.as_f64()?;
            (f.fract() == 0.0 && f.abs() < 9e15).then_some(f as i64)
        }),
        _ => None,
    }
}

fn want_opt_int(m: &Map<String, Value>, k: &str) -> std::result::Result<Option<i64>, String> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => as_int(v)
            .map(Some)
            .ok_or_else(|| format!("{k}: expected an integer")),
    }
}

/// `cooldown_s` and `fires`: `None` is zero (the model's `_none_is_zero`).
fn want_int0(m: &Map<String, Value>, k: &str) -> std::result::Result<i64, String> {
    Ok(want_opt_int(m, k)?.unwrap_or(0))
}

fn want_dt(m: &Map<String, Value>, k: &str) -> std::result::Result<Option<Dt>, String> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Dt::parse(s)
            .map(Some)
            .ok_or_else(|| format!("{k}: invalid datetime {s:?}")),
        Some(_) => Err(format!("{k}: expected a datetime string")),
    }
}

impl Intention {
    /// A new pending intention with Python's defaults.
    pub fn new(id: &str, kind: Kind, title: &str) -> Intention {
        Intention {
            id: id.into(),
            kind,
            title: title.into(),
            state: State::Pending,
            created_at: Dt::now(),
            origin_session: None,
            origin_memory_id: None,
            channel: "inline".into(),
            due_at: None,
            snooze_until: None,
            interval_s: None,
            trigger_terms: Vec::new(),
            scope: Vec::new(),
            cooldown_s: 0,
            max_fires: None,
            fires: 0,
            last_fired_at: None,
            expires_at: None,
            extra: Map::new(),
        }
    }

    /// Validate a JSON object as `Intention(**data)` does.
    pub fn from_value(v: &Value) -> std::result::Result<Intention, String> {
        let m = v.as_object().ok_or("not a JSON object")?;
        let id = want_str(m, "id")?.ok_or("id: required")?;
        let kind = match want_str(m, "kind")?.ok_or("kind: required")?.as_str() {
            "time" => Kind::Time,
            "event" => Kind::Event,
            o => return Err(format!("kind: invalid {o:?}")),
        };
        let title = want_str(m, "title")?.ok_or("title: required")?;
        let state = match m.get("state") {
            None => State::Pending,
            Some(Value::String(s)) => State::parse(s).ok_or(format!("state: invalid {s:?}"))?,
            Some(_) => return Err("state: expected a string".into()),
        };
        let created_at = match m.get("created_at") {
            None => Dt::now(),
            Some(Value::String(s)) => {
                Dt::parse(s).ok_or(format!("created_at: invalid datetime {s:?}"))?
            }
            Some(_) => return Err("created_at: expected a datetime string".into()),
        };
        let channel = want_str(m, "channel")?.unwrap_or_else(|| "inline".into());
        let trigger_terms = match m.get("trigger_terms") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_string)
                        .ok_or("trigger_terms: expected strings".to_string())
                })
                .collect::<std::result::Result<_, _>>()?,
            Some(_) => return Err("trigger_terms: expected a list".into()),
        };
        let mut scope: Vec<(String, String)> = Vec::new();
        match m.get("scope") {
            None => {}
            Some(Value::Object(o)) => {
                for (k, x) in o {
                    let s = x.as_str().ok_or("scope: expected string values")?;
                    scope.push((k.clone(), s.to_string()));
                }
            }
            Some(_) => return Err("scope: expected an object".into()),
        }
        let extra: Map<String, Value> = m
            .iter()
            .filter(|(k, _)| !DECLARED.contains(&k.as_str()))
            .map(|(k, x)| (k.clone(), x.clone()))
            .collect();
        Ok(Intention {
            id,
            kind,
            title,
            state,
            created_at,
            origin_session: want_opt_str(m, "origin_session")?,
            origin_memory_id: want_opt_str(m, "origin_memory_id")?,
            channel,
            due_at: want_dt(m, "due_at")?,
            snooze_until: want_dt(m, "snooze_until")?,
            interval_s: want_opt_int(m, "interval_s")?,
            trigger_terms,
            scope,
            cooldown_s: want_int0(m, "cooldown_s")?,
            max_fires: want_opt_int(m, "max_fires")?,
            fires: want_int0(m, "fires")?,
            last_fired_at: want_dt(m, "last_fired_at")?,
            expires_at: want_dt(m, "expires_at")?,
            extra,
        })
    }

    /// Parse a flat file's text.
    pub fn from_json(text: &str) -> std::result::Result<Intention, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        Intention::from_value(&v)
    }

    /// The model as a JSON value in pydantic field order, extras last.
    pub fn to_value(&self) -> Value {
        let ds = |d: &Option<Dt>| d.map_or(Value::Null, |x| Value::String(x.pydantic()));
        let os = |s: &Option<String>| s.clone().map_or(Value::Null, Value::String);
        let oi = |n: Option<i64>| n.map_or(Value::Null, Value::from);
        let mut m = Map::new();
        m.insert("id".into(), self.id.clone().into());
        m.insert("kind".into(), self.kind.as_str().into());
        m.insert("title".into(), self.title.clone().into());
        m.insert("state".into(), self.state.as_str().into());
        m.insert("created_at".into(), self.created_at.pydantic().into());
        m.insert("origin_session".into(), os(&self.origin_session));
        m.insert("origin_memory_id".into(), os(&self.origin_memory_id));
        m.insert("channel".into(), self.channel.clone().into());
        m.insert("due_at".into(), ds(&self.due_at));
        m.insert("snooze_until".into(), ds(&self.snooze_until));
        m.insert("interval_s".into(), oi(self.interval_s));
        m.insert(
            "trigger_terms".into(),
            Value::Array(
                self.trigger_terms
                    .iter()
                    .map(|t| t.clone().into())
                    .collect(),
            ),
        );
        let mut sc = Map::new();
        for (k, v) in &self.scope {
            sc.insert(k.clone(), v.clone().into());
        }
        m.insert("scope".into(), Value::Object(sc));
        m.insert("cooldown_s".into(), self.cooldown_s.into());
        m.insert("max_fires".into(), oi(self.max_fires));
        m.insert("fires".into(), self.fires.into());
        m.insert("last_fired_at".into(), ds(&self.last_fired_at));
        m.insert("expires_at".into(), ds(&self.expires_at));
        for (k, v) in &self.extra {
            m.insert(k.clone(), v.clone());
        }
        Value::Object(m)
    }

    /// `model_dump_json(indent=2)`: the text written to the flat file.
    pub fn dump_indent2(&self) -> String {
        pyd_dump(&self.to_value(), Some(2))
    }

    /// `model_dump_json()`: compact, used as the merge tie-break.
    pub fn dump_compact(&self) -> String {
        pyd_dump(&self.to_value(), None)
    }
}

// ---------------------------------------------------------------------------
// transitions and merge
// ---------------------------------------------------------------------------

/// Every event name the table knows.
pub const EVENTS: [&str; 7] = [
    "arm",
    "fire",
    "done",
    "snooze",
    "cancel",
    "expire",
    "invalidate",
];

fn next_state(state: State, event: &str) -> Option<State> {
    use State::*;
    Some(match (state, event) {
        (Pending, "arm") => Armed,
        (Pending, "fire") => Fired,
        (Pending, "done") => Done,
        (Pending, "snooze") => Pending,
        (Pending, "cancel") => Cancelled,
        (Pending, "expire") => Expired,
        (Pending, "invalidate") => Invalidated,
        (Armed, "fire") => Fired,
        (Armed, "snooze") => Armed,
        (Armed, "done") => Done,
        (Armed, "cancel") => Cancelled,
        (Armed, "expire") => Expired,
        (Armed, "invalidate") => Invalidated,
        (Fired, "fire") => Fired,
        (Fired, "snooze") => Armed,
        (Fired, "done") => Done,
        (Fired, "cancel") => Cancelled,
        (Fired, "expire") => Expired,
        (Fired, "invalidate") => Invalidated,
        _ => return None,
    })
}

fn known_event(event: &str) -> bool {
    EVENTS.contains(&event)
}

fn repeats(i: &Intention, fires_after: i64) -> bool {
    i.kind == Kind::Time
        && i.interval_s.is_some_and(|s| s > 0)
        && i.due_at.is_some()
        && i.max_fires.is_none_or(|m| fires_after < m)
}

/// The intention after `event`; the input is never mutated. Terminal states
/// ignore every event; a non-terminal state with no row for the event (or an
/// unknown event) is an error.
pub fn transition(
    i: &Intention,
    event: &str,
    now: Option<Dt>,
    snooze_until: Option<Dt>,
) -> Result<Intention> {
    if !known_event(event) {
        return verr(format!("unknown intention event: {}", rp(event)));
    }
    if i.state.is_terminal() {
        return Ok(i.clone());
    }
    let Some(nxt) = next_state(i.state, event) else {
        return verr(format!(
            "no transition from {} on {}",
            rp(i.state.as_str()),
            rp(event)
        ));
    };
    let mut o = i.clone();
    o.state = nxt;
    if event == "fire" {
        o.fires = i.fires + 1;
        let fired_at = now.unwrap_or_else(Dt::now);
        o.last_fired_at = Some(fired_at);
        if repeats(i, o.fires) {
            // Advance by whole intervals to the first slot strictly after now,
            // so a long outage yields one fire, not a storm of catch-up fires.
            let step = i.interval_s.unwrap_or(1) * 1_000_000;
            let due = i.due_at.unwrap_or(fired_at);
            let k = ((fired_at.micros - due.micros).div_euclid(step) + 1).max(1);
            o.state = State::Armed;
            o.due_at = Some(due.add_micros(step * k));
            o.snooze_until = None;
        }
    } else if event == "snooze" {
        if let Some(s) = snooze_until {
            o.snooze_until = Some(s);
        }
    }
    Ok(o)
}

/// Deterministic, commutative merge of two replicas of one intention.
/// The winner by `(state rank, fires, last_fired_at, compact JSON)` supplies
/// the record; `fires` and `last_fired_at` are the maxima of both.
pub fn merge_intentions(a: &Intention, b: &Intention) -> Intention {
    let key = |i: &Intention| {
        (
            i.state.rank(),
            i.fires,
            i.last_fired_at.unwrap_or(EPOCH),
            i.dump_compact(),
        )
    };
    let win = if key(b) > key(a) { b } else { a };
    let mut out = win.clone();
    out.fires = a.fires.max(b.fires);
    out.last_fired_at = match (a.last_fired_at, b.last_fired_at) {
        (Some(x), Some(y)) => Some(if y > x { y } else { x }),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    };
    out
}

// ---------------------------------------------------------------------------
// ids, due_at parsing
// ---------------------------------------------------------------------------

/// `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`, including Python's `$` that also
/// matches before one trailing newline.
pub fn id_ok(id: &str) -> bool {
    let t = id.strip_suffix('\n').unwrap_or(id);
    let b = t.as_bytes();
    !b.is_empty()
        && b.len() <= 128
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-'))
}

fn check_id(id: &str) -> Result<&str> {
    if id_ok(id) {
        Ok(id)
    } else {
        verr(format!("invalid intention id: {}", rp(id)))
    }
}

/// `parse_due_at`: an ISO 8601 timestamp or `in N minutes|hours|days|weeks`,
/// relative forms resolved against `now`. Never guesses.
pub fn parse_due_at(value: &str, now: Option<Dt>) -> Result<Dt> {
    let now = now.unwrap_or_else(Dt::now);
    if value.trim().is_empty() {
        return verr("due_at is required: an ISO 8601 time or 'in N days/hours'");
    }
    if let Some(d) = parse_relative(value) {
        return Ok(now.add_micros(d));
    }
    match Dt::parse(py_strip(value)) {
        Some(d) => Ok(d),
        None => verr(format!(
            "unrecognised due_at {}: use an ISO 8601 time (2026-12-01T09:00:00Z) or 'in N days/hours/minutes/weeks'",
            rp(value)
        )),
    }
}

fn parse_relative(value: &str) -> Option<i64> {
    let t = py_strip(value).to_lowercase();
    let rest = t.strip_prefix("in")?;
    let rest2 = rest.trim_start_matches(|c: char| c.is_whitespace());
    if rest2.len() == rest.len() {
        return None;
    }
    let nd = rest2.bytes().take_while(u8::is_ascii_digit).count();
    if nd == 0 || nd > 6 {
        return None;
    }
    let n: i64 = rest2[..nd].parse().ok()?;
    let unit = rest2[nd..].trim_start_matches(|c: char| c.is_whitespace());
    let secs = match unit {
        "minute" | "minutes" | "min" | "mins" => 60,
        "hour" | "hours" | "hr" | "hrs" => 3600,
        "day" | "days" => 86_400,
        "week" | "weeks" => 604_800,
        _ => return None,
    };
    Some(n * secs * 1_000_000)
}

// ---------------------------------------------------------------------------
// prefilter helpers
// ---------------------------------------------------------------------------

fn is_word(c: char) -> bool {
    is_py_word_char(c)
}

/// Maximal `\w+` runs of `s`.
fn word_runs(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in s.char_indices() {
        if is_word(c) {
            start.get_or_insert(i);
        } else if let Some(st) = start.take() {
            out.push(&s[st..i]);
        }
    }
    if let Some(st) = start {
        out.push(&s[st..]);
    }
    out
}

/// Phrase hit with word boundaries only where the term edge is a word char:
/// `fusion` must not match inside `confusion`, while `HOSTS=` ends in a
/// non-word char and needs no guard there. Multi-word terms match only with
/// exactly the term's own spacing.
fn term_hit(term: &str, low: &str) -> bool {
    let Some(first) = term.chars().next() else {
        return false;
    };
    let last = term.chars().next_back().unwrap_or(first);
    let (pre, post) = (is_word(first), is_word(last));
    let mut from = 0;
    while let Some(p) = low[from..].find(term) {
        let at = from + p;
        let end = at + term.len();
        let before_ok = !pre || !low[..at].chars().next_back().is_some_and(is_word);
        let after_ok = !post || !low[end..].chars().next().is_some_and(is_word);
        if before_ok && after_ok {
            return true;
        }
        from = at + low[at..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

fn phrase_hit(terms: &[String], low: &str) -> bool {
    terms.iter().any(|t| !t.is_empty() && term_hit(t, low))
}

fn scope_ok(want: &[(String, String)], have: &HashMap<String, String>) -> bool {
    want.iter().all(|(k, v)| have.get(k) == Some(v))
}

#[allow(clippy::too_many_arguments)]
fn gate_ok(
    state: &str,
    fires: i64,
    max_fires: Option<i64>,
    cooldown_s: i64,
    last: Option<Dt>,
    expires: Option<Dt>,
    now: Dt,
) -> bool {
    if state != "armed" && state != "fired" {
        return false;
    }
    if expires.is_some_and(|e| e <= now) {
        return false;
    }
    if max_fires.is_some_and(|m| fires >= m) {
        return false;
    }
    !(cooldown_s != 0
        && last.is_some_and(|l| ((now.micros - l.micros) as f64 / 1e6) < cooldown_s as f64))
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(format!("SKMEMORY_PREFILTER_{name}"))
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Options of one [`IntentionStore::prefilter`] call.
#[derive(Clone, Debug, Default)]
pub struct PrefilterOpts {
    /// Whole-call deadline in seconds (default env or 50 ms).
    pub deadline_s: Option<f64>,
    /// Fire cap (default env or 3).
    pub max_fires: Option<i64>,
    /// Start of the deadline window (default: entry).
    pub t0: Option<Instant>,
}

/// Cheap change key of the index: write count plus db and WAL stat.
type Stamp = (u64, [(u128, u64); 2]);

#[derive(Default, Clone)]
struct Matcher {
    by_tok: HashMap<String, Vec<String>>,
    nonword: Vec<String>,
    terms_of: HashMap<String, Vec<String>>,
}

/// What a reindex did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReindexReport {
    pub indexed: usize,
    /// `(id, conflict file names merged)`.
    pub merged: Vec<(String, Vec<String>)>,
    pub conflicts: Vec<String>,
    pub unreadable: Vec<String>,
}

const BASE_COLS: [&str; 9] = [
    "id",
    "kind",
    "state",
    "due_at",
    "snooze_until",
    "expires_at",
    "origin_memory_id",
    "channel",
    "data",
];
const PREFILTER_COLS: [(&str, &str); 6] = [
    ("terms", "TEXT"),
    ("scope", "TEXT"),
    ("cooldown_s", "INTEGER"),
    ("max_fires", "INTEGER"),
    ("fires", "INTEGER"),
    ("last_fired_at", "TEXT"),
];

fn upsert_sql() -> String {
    let cols: Vec<&str> = BASE_COLS
        .iter()
        .copied()
        .chain(PREFILTER_COLS.iter().map(|c| c.0))
        .collect();
    format!(
        "INSERT OR REPLACE INTO intentions ({}) VALUES ({})",
        cols.join(","),
        vec!["?"; cols.len()].join(",")
    )
}

fn tmp_root() -> PathBuf {
    // tempfile.gettempdir(): TMPDIR, TEMP, TMP, then the usual directories,
    // first one a file can be created in.
    let mut cands: Vec<PathBuf> = ["TMPDIR", "TEMP", "TMP"]
        .iter()
        .filter_map(std::env::var_os)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .collect();
    cands.extend(["/tmp", "/var/tmp", "/usr/tmp"].map(PathBuf::from));
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd);
    }
    for c in cands {
        let probe = c.join(format!(".skmem-probe-{}", std::process::id()));
        if OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .is_ok()
        {
            let _ = fs::remove_file(&probe);
            return c;
        }
    }
    PathBuf::from("/tmp")
}

/// Host-local, never-synced directory root for lock files: `$XDG_RUNTIME_DIR`
/// (a per-user tmpfs) when set, else the system temp dir. Lock files must not
/// live in the synced memory dir.
pub fn lock_root() -> PathBuf {
    let root = std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(tmp_root);
    root.join(format!("skmemory-intent-locks-{}", sys::uid()))
}

/// The lock directory for a store at `base`:
/// `<lock root>/skmemory-intent-locks-<uid>/<sha1(str(base.resolve()))[:16]>`.
/// `lock_root_dir` overrides the root (default: `$XDG_RUNTIME_DIR`, else the temp dir).
pub fn lock_dir_for(base: &Path, lock_root_dir: Option<&Path>) -> PathBuf {
    let root = match lock_root_dir {
        Some(r) => r.join(format!("skmemory-intent-locks-{}", sys::uid())),
        None => lock_root(),
    };
    let digest = sys::sha1_hex(resolve(base).to_string_lossy().as_bytes());
    root.join(&digest[..16])
}

/// `Path(p).resolve()` for a path that exists (non-strict otherwise).
fn resolve(p: &Path) -> PathBuf {
    if let Ok(c) = fs::canonicalize(p) {
        return c;
    }
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// An exclusive intention lock; released on drop.
pub struct LockGuard {
    file: Option<File>,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(f) = &self.file {
            let _ = sys::unlock(f);
        }
    }
}

/// A non-blocking lock could not be taken in time.
#[derive(Debug, PartialEq, Eq)]
pub struct LockBusy(pub String);

fn mtime_ns(p: &Path) -> (u128, u64) {
    fs::metadata(p)
        .ok()
        .map(|m| {
            let t = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            (t, m.len())
        })
        .unwrap_or((0, 0))
}

fn q(c: &Connection, sql: &str) -> rusqlite::Result<()> {
    c.execute_batch(sql)
}

// ---------------------------------------------------------------------------
// the store
// ---------------------------------------------------------------------------

/// Flat-file store (source of truth) with a derived `intentions` table in `index.db`.
pub struct IntentionStore {
    base: PathBuf,
    dir: PathBuf,
    db_path: PathBuf,
    conn: Connection,
    gen: Cell<u64>,
    pf_cache: RefCell<Option<(Stamp, Rc<Matcher>)>>,
    lock_dir: PathBuf,
    /// Unreadable files or rows skipped by this store instance.
    pub unreadable: Cell<u64>,
    /// What the last [`IntentionStore::reindex`] did.
    pub last_reindex_report: RefCell<ReindexReport>,
}

impl IntentionStore {
    /// Open with the default 30 s busy timeout and the default lock root.
    pub fn open(base: impl AsRef<Path>, owner: &OwnerScope) -> Result<IntentionStore> {
        Self::open_with(base, 30_000, None, owner)
    }

    /// Open with an explicit SQLite busy timeout (ms) and, for tests that
    /// must not touch the process environment, an explicit lock root (the
    /// directory that would hold `skmemory-intent-locks-<uid>`).
    pub fn open_with(
        base: impl AsRef<Path>,
        busy_ms: u64,
        lock_root_dir: Option<&Path>,
        _owner: &OwnerScope,
    ) -> Result<IntentionStore> {
        let base = base.as_ref().to_path_buf();
        let dir = base.join("intentions");
        fs::create_dir_all(&dir)?;
        let db_path = base.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.busy_timeout(Duration::from_millis(busy_ms))?;
        let _ = conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()));
        crate::index::wal_apply(&conn);
        let lock_dir = lock_dir_for(&base, lock_root_dir);
        let st = IntentionStore {
            lock_dir,
            base,
            dir,
            db_path,
            conn,
            gen: Cell::new(0),
            pf_cache: RefCell::new(None),
            unreadable: Cell::new(0),
            last_reindex_report: RefCell::new(ReindexReport::default()),
        };
        st.ensure_table()?;
        Ok(st)
    }

    /// The memory base directory.
    pub fn base_path(&self) -> &Path {
        &self.base
    }

    /// The directory holding the flat `<id>.json` files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The host-local lock directory shared with the Python store.
    pub fn lock_dir(&self) -> &Path {
        &self.lock_dir
    }

    fn ensure_table(&self) -> Result<()> {
        q(
            &self.conn,
            "CREATE TABLE IF NOT EXISTS intentions (id TEXT PRIMARY KEY, kind TEXT, state TEXT, \
             due_at TEXT, snooze_until TEXT, expires_at TEXT, origin_memory_id TEXT, channel TEXT, \
             data TEXT NOT NULL)",
        )?;
        let mut have = BTreeSet::new();
        {
            let mut st = self.conn.prepare("PRAGMA table_info(intentions)")?;
            for r in st.query_map([], |r| r.get::<_, String>(1))? {
                have.insert(r?);
            }
        }
        let mut added = false;
        for (col, typ) in PREFILTER_COLS {
            if !have.contains(col) {
                // A concurrent opener (Python or Rust) may add it first.
                match q(
                    &self.conn,
                    &format!("ALTER TABLE intentions ADD COLUMN {col} {typ}"),
                ) {
                    Ok(()) => added = true,
                    Err(e) if e.to_string().to_lowercase().contains("duplicate column") => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        q(
            &self.conn,
            "CREATE INDEX IF NOT EXISTS idx_intentions_state ON intentions(state)",
        )?;
        if added {
            self.backfill_prefilter_cols()?;
        }
        Ok(())
    }

    fn backfill_prefilter_cols(&self) -> Result<()> {
        let rows: Vec<(String, String)> = {
            let mut st = self.conn.prepare("SELECT id, data FROM intentions")?;
            let it = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            it.filter_map(std::result::Result::ok).collect()
        };
        let tx = self.conn.unchecked_transaction()?;
        for (id, data) in rows {
            let Ok(i) = Intention::from_json(&data) else {
                continue;
            };
            let mut vals = prefilter_vals(&i);
            vals.push(SqlValue::Text(id));
            tx.execute(
                "UPDATE intentions SET terms=?, scope=?, cooldown_s=?, max_fires=?, fires=?, \
                 last_fired_at=? WHERE id=?",
                params_from_iter(vals),
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn index(&self, i: &Intention, body: &str) -> Result<()> {
        self.gen.set(self.gen.get() + 1);
        self.conn
            .execute(&upsert_sql(), params_from_iter(row_values(i, body)))?;
        Ok(())
    }

    // -- CRUD -----------------------------------------------------------

    fn path(&self, id: &str) -> Result<PathBuf> {
        Ok(self.dir.join(format!("{}.json", check_id(id)?)))
    }

    fn unreadable_hit(&self, p: &Path, err: &str) {
        self.unreadable.set(self.unreadable.get() + 1);
        warn(&format!(
            "unreadable intention file {}: {err}",
            p.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        ));
    }

    /// Read the flat file. Missing is `None`; corrupt is `None`, counted and logged.
    fn read(&self, id: &str) -> Result<Option<Intention>> {
        let p = self.path(id)?;
        let raw = match fs::read(&p) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                self.unreadable_hit(&p, &e.to_string());
                return Ok(None);
            }
        };
        let text = match String::from_utf8(raw) {
            Ok(t) => t,
            Err(e) => {
                self.unreadable_hit(&p, &e.to_string());
                return Ok(None);
            }
        };
        match Intention::from_json(&text) {
            Ok(i) => Ok(Some(i)),
            Err(e) => {
                self.unreadable_hit(&p, &e);
                Ok(None)
            }
        }
    }

    /// Exclusive flock on the per-intention, host-local lock file (the same
    /// file Python's `_lock` takes). `timeout` of `None` blocks; `Some(secs)`
    /// polls non-blocking and returns `Err(LockBusy)` on expiry (`Some(0.0)`
    /// is a single try). If locking is unavailable (unwritable lock dir, flock
    /// error) this warns and proceeds unlocked, as Python does.
    pub fn lock(&self, id: &str, timeout: Option<f64>) -> std::result::Result<LockGuard, LockBusy> {
        let name = format!("{}.lock", check_id(id).map_err(|_| LockBusy(id.into()))?);
        self.lock_named(&name, timeout)
    }

    /// Like [`IntentionStore::lock`] for an arbitrary lock file name in the
    /// lock directory (the tick takes `_tick.lock`).
    pub fn lock_named(
        &self,
        name: &str,
        timeout: Option<f64>,
    ) -> std::result::Result<LockGuard, LockBusy> {
        let setup = || -> std::io::Result<File> {
            fs::create_dir_all(&self.lock_dir)?;
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .mode(0o600)
                .open(self.lock_dir.join(name))
        };
        let file = match setup() {
            Ok(f) => f,
            Err(e) => {
                warn(&format!(
                    "intention lock unavailable for {name}, proceeding unlocked: {e}"
                ));
                return Ok(LockGuard { file: None });
            }
        };
        match timeout {
            None => {
                if let Err(e) = sys::lock_exclusive(&file) {
                    warn(&format!(
                        "intention lock unavailable for {name}, proceeding unlocked: {e}"
                    ));
                    return Ok(LockGuard { file: None });
                }
            }
            Some(t) => {
                let deadline = Instant::now() + Duration::from_secs_f64(t.max(0.0));
                loop {
                    match sys::try_lock_exclusive(&file) {
                        Ok(sys::TryLock::Acquired) => break,
                        Ok(sys::TryLock::Busy) => {
                            if Instant::now() >= deadline {
                                return Err(LockBusy(name.to_string()));
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => {
                            warn(&format!(
                                "intention lock unavailable for {name}, proceeding unlocked: {e}"
                            ));
                            return Ok(LockGuard { file: None });
                        }
                    }
                }
            }
        }
        Ok(LockGuard { file: Some(file) })
    }

    fn write(&self, i: &Intention) -> Result<()> {
        let body = i.dump_indent2();
        atomic_write_bytes(&self.path(&i.id)?, body.as_bytes())?;
        self.index(i, &body)
    }

    /// Create an intention, or rewrite one whose state is unchanged on disk.
    /// A save that would change the state of an existing intention is refused
    /// (use [`IntentionStore::apply`]), so a dead intention cannot be revived
    /// by writing a stale copy. Flat file first, then index.
    pub fn save(&self, i: &Intention) -> Result<()> {
        let _g = self
            .lock(&i.id, None)
            .map_err(|_| IntentionError::Value(format!("invalid intention id: {}", rp(&i.id))))?;
        if let Some(on_disk) = self.read(&i.id)? {
            if on_disk.state != i.state {
                return verr(format!(
                    "refusing state change {} -> {} for {} via save(); use apply() so transition() decides",
                    rp(on_disk.state.as_str()),
                    rp(i.state.as_str()),
                    rp(&i.id)
                ));
            }
        }
        self.write(i)
    }

    /// Re-read the flat file, apply [`transition`], write. The only
    /// state-change path. `None` if the file is missing or unreadable.
    pub fn apply(
        &self,
        id: &str,
        event: &str,
        now: Option<Dt>,
        snooze_until: Option<Dt>,
    ) -> Result<Option<Intention>> {
        check_id(id)?;
        let _g = self
            .lock(id, None)
            .map_err(|_| IntentionError::Value(format!("invalid intention id: {}", rp(id))))?;
        let Some(cur) = self.read(id)? else {
            return Ok(None);
        };
        let new = transition(&cur, event, now, snooze_until)?;
        if new != cur {
            self.write(&new)?;
        }
        Ok(Some(new))
    }

    /// Read from the flat file (source of truth); a corrupt file reads as `None`.
    pub fn get(&self, id: &str) -> Result<Option<Intention>> {
        self.read(id)
    }

    /// List from the index, skipping unparseable rows. Run
    /// [`IntentionStore::reindex`] after a sync.
    pub fn list(&self, state: Option<&str>) -> Result<Vec<Intention>> {
        let (sql, arg) = match state.filter(|s| !s.is_empty()) {
            Some(s) => (
                "SELECT id, data FROM intentions WHERE state=? ORDER BY id",
                Some(s.to_string()),
            ),
            None => ("SELECT id, data FROM intentions ORDER BY id", None),
        };
        let mut st = self.conn.prepare(sql)?;
        let map = |r: &rusqlite::Row<'_>| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?));
        let rows: Vec<(String, String)> = match arg {
            Some(a) => st
                .query_map([a], map)?
                .collect::<std::result::Result<_, _>>()?,
            None => st
                .query_map([], map)?
                .collect::<std::result::Result<_, _>>()?,
        };
        let mut out = Vec::new();
        for (id, data) in rows {
            match Intention::from_json(&data) {
                Ok(i) => out.push(i),
                Err(e) => {
                    self.unreadable.set(self.unreadable.get() + 1);
                    warn(&format!("unreadable intention index row {id}: {e}"));
                }
            }
        }
        Ok(out)
    }

    fn json_files(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = match fs::read_dir(&self.dir) {
            Ok(rd) => rd
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
                .map(|e| e.path())
                .collect(),
            Err(_) => Vec::new(),
        };
        v.sort();
        v
    }

    fn parse_file(p: &Path) -> std::result::Result<Intention, String> {
        let raw = fs::read(p).map_err(|e| e.to_string())?;
        let text = String::from_utf8(raw).map_err(|e| e.to_string())?;
        Intention::from_json(&text)
    }

    /// Rebuild the table from the flat files in one transaction; returns rows
    /// indexed. Syncthing conflict copies (any file whose stem is not the id
    /// inside) are merged with [`merge_intentions`], the result is written
    /// atomically to the canonical `<id>.json`, and only then are the copies
    /// removed. A conflict-only id is promoted to canonical. Deterministic, so
    /// two hosts converge.
    pub fn reindex(&self) -> Result<usize> {
        let mut groups: Vec<(String, Vec<PathBuf>)> = Vec::new();
        let mut bad: Vec<String> = Vec::new();
        for p in self.json_files() {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match Self::parse_file(&p) {
                Ok(i) if id_ok(&i.id) => match groups.iter_mut().find(|g| g.0 == i.id) {
                    Some(g) => g.1.push(p),
                    None => groups.push((i.id.clone(), vec![p])),
                },
                Ok(i) => {
                    warn(&format!(
                        "skipping unreadable intention file {name}: invalid id {:?}",
                        i.id
                    ));
                    bad.push(name);
                }
                Err(e) => {
                    warn(&format!("skipping unreadable intention file {name}: {e}"));
                    bad.push(name);
                }
            }
        }
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut merged: Vec<(String, Vec<String>)> = Vec::new();
        for (iid, paths) in &groups {
            let stem_of = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().into_owned());
            if paths.len() == 1 && stem_of(&paths[0]).as_deref() == Some(iid.as_str()) {
                if let Ok(i) = Self::parse_file(&paths[0]) {
                    let body = i.dump_indent2();
                    rows.push(row_values(&i, &body));
                }
                continue;
            }
            let res = {
                let _g = self
                    .lock(iid, None)
                    .map_err(|_| IntentionError::Value("lock".into()))?;
                self.merge_group(iid)?
            };
            if let Some(res) = res {
                let files = paths
                    .iter()
                    .filter(|p| stem_of(p).as_deref() != Some(iid.as_str()))
                    .map(|p| {
                        p.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    })
                    .collect();
                merged.push((iid.clone(), files));
                let body = res.dump_indent2();
                rows.push(row_values(&res, &body));
            }
        }
        self.gen.set(self.gen.get() + 1);
        {
            let tx = self.conn.unchecked_transaction()?;
            tx.execute("DELETE FROM intentions", [])?;
            let sql = upsert_sql();
            for r in rows.iter() {
                tx.execute(&sql, params_from_iter(r.iter()))?;
            }
            tx.commit()?;
        }
        self.unreadable
            .set(self.unreadable.get() + bad.len() as u64);
        let conflicts = merged.iter().flat_map(|m| m.1.clone()).collect();
        let n = rows.len();
        *self.last_reindex_report.borrow_mut() = ReindexReport {
            indexed: n,
            merged,
            conflicts,
            unreadable: bad,
        };
        Ok(n)
    }

    /// Merge canonical plus conflict copies of `iid` (caller holds the lock).
    /// Re-reads every file under the lock. The canonical file is rewritten only
    /// if the merge differs from it; copies are unlinked only after that write.
    fn merge_group(&self, iid: &str) -> Result<Option<Intention>> {
        let canon = self.path(iid)?;
        let mut replicas: Vec<Intention> = Vec::new();
        let mut copies: Vec<PathBuf> = Vec::new();
        let mut canon_val: Option<Intention> = None;
        for p in self.json_files() {
            let Ok(i) = Self::parse_file(&p) else {
                continue;
            };
            if i.id != iid {
                continue;
            }
            if p == canon {
                canon_val = Some(i.clone());
            } else {
                copies.push(p);
            }
            replicas.push(i);
        }
        let mut it = replicas.into_iter();
        let Some(mut result) = it.next() else {
            return Ok(None);
        };
        for r in it {
            result = merge_intentions(&result, &r);
        }
        if canon_val.as_ref() != Some(&result) {
            self.write(&result)?;
        }
        for p in copies {
            warn(&format!(
                "merged intention conflict copy {} into {iid}.json",
                p.file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ));
            let _ = fs::remove_file(&p);
        }
        Ok(Some(result))
    }

    // -- prefilter ------------------------------------------------------

    fn stamp(&self) -> Stamp {
        let wal = {
            let mut s = self.db_path.as_os_str().to_owned();
            s.push("-wal");
            PathBuf::from(s)
        };
        (self.gen.get(), [mtime_ns(&self.db_path), mtime_ns(&wal)])
    }

    /// First-token index of armed or fired event intentions, cached on the
    /// index stamp. Every term with a word char starts at a whole token of the
    /// prompt, so one tokenisation plus lookups finds every candidate and the
    /// phrase check only verifies those. Terms with no word char are always verified.
    fn matcher(&self) -> Result<Rc<Matcher>> {
        let key = self.stamp();
        if let Some((k, m)) = &*self.pf_cache.borrow() {
            if *k == key {
                return Ok(Rc::clone(m));
            }
        }
        let mut m = Matcher::default();
        let mut st = self.conn.prepare(
            "SELECT id, terms FROM intentions WHERE kind='event' AND state IN ('armed','fired')",
        )?;
        let rows = st.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
        })?;
        for r in rows {
            let (iid, terms) = r?;
            let Some(terms) = terms.filter(|t| !t.is_empty()) else {
                continue;
            };
            let ts: Vec<String> = terms
                .split(SEP)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect();
            for t in &ts {
                match word_runs(t).first() {
                    None => m.nonword.push(iid.clone()),
                    Some(tok) => m
                        .by_tok
                        .entry((*tok).to_string())
                        .or_default()
                        .push(iid.clone()),
                }
            }
            m.terms_of.insert(iid, ts);
        }
        let m = Rc::new(m);
        *self.pf_cache.borrow_mut() = Some((key, Rc::clone(&m)));
        Ok(m)
    }

    /// Fire and return the event intentions this prompt triggers.
    ///
    /// Case-insensitive, word-bounded phrase match of any trigger term (no
    /// stemming); every key of the intention's scope must equal the prompt
    /// scope; skipped when not armed or fired, expired, at max_fires, or inside
    /// the cooldown. Candidates come from the index, then each is re-checked
    /// and fired on the flat file under its lock via [`transition`], so a stale
    /// index cannot double-fire and flat and index move together.
    ///
    /// Bounded work: one deadline for the whole call (default 50 ms), at most
    /// `max_fires` (default 3) intentions fire, the fire path only try-locks
    /// (a locked intention is skipped this prompt), and only the head and tail
    /// of `SKMEMORY_PREFILTER_MAX_CHARS` (default 16384) of a long prompt is
    /// scanned; a token split by the cut is dropped, so `fusion` never fires on
    /// the tail of `confusion`. Multi-word terms need exactly one space between
    /// words.
    pub fn prefilter(
        &self,
        text: &str,
        scope: &HashMap<String, String>,
        now: Option<Dt>,
        opts: &PrefilterOpts,
    ) -> Result<Vec<Intention>> {
        let t0 = opts.t0.unwrap_or_else(Instant::now);
        let deadline = opts
            .deadline_s
            .unwrap_or_else(|| env_num("DEADLINE_S", DEFAULT_DEADLINE_S));
        let end = t0 + Duration::from_secs_f64(deadline.clamp(0.0, 1e9));
        let cap = opts
            .max_fires
            .unwrap_or_else(|| env_num("MAX_FIRES", DEFAULT_MAX_FIRES));
        let limit = env_num("MAX_CHARS", DEFAULT_MAX_CHARS);
        let now = now.unwrap_or_else(Dt::now);
        let low = scan_text(text, limit).to_lowercase();
        if py_strip(&low).is_empty() || cap <= 0 {
            return Ok(Vec::new());
        }
        let m = self.matcher()?;
        let mut ids: BTreeSet<&str> = m.nonword.iter().map(String::as_str).collect();
        let toks: HashSet<&str> = word_runs(&low).into_iter().collect();
        for tok in toks {
            if let Some(hit) = m.by_tok.get(tok) {
                ids.extend(hit.iter().map(String::as_str));
            }
        }
        let mut fired = Vec::new();
        for iid in ids {
            if Instant::now() >= end || fired.len() as i64 >= cap {
                break;
            }
            let Some(terms) = m.terms_of.get(iid) else {
                continue;
            };
            if !phrase_hit(terms, &low) {
                continue;
            }
            type Gate = (
                Option<String>,
                Option<i64>,
                Option<i64>,
                Option<i64>,
                Option<String>,
                Option<String>,
                Option<String>,
            );
            let row: Option<Gate> = self
                .conn
                .query_row(
                    "SELECT scope, cooldown_s, max_fires, fires, last_fired_at, expires_at, state \
                     FROM intentions WHERE id=?",
                    [iid],
                    |r| {
                        Ok((
                            r.get(0)?,
                            r.get(1)?,
                            r.get(2)?,
                            r.get(3)?,
                            r.get(4)?,
                            r.get(5)?,
                            r.get(6)?,
                        ))
                    },
                )
                .ok();
            let Some((sc, cool, maxf, fires, last, exp, state)) = row else {
                continue;
            };
            let Ok(Value::Object(o)) = serde_json::from_str::<Value>(
                sc.as_deref().filter(|s| !s.is_empty()).unwrap_or("{}"),
            ) else {
                continue;
            };
            // A non-string scope value can never equal a prompt scope string.
            let Some(want) = o
                .into_iter()
                .map(|(k, x)| x.as_str().map(|s| (k, s.to_string())))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if !scope_ok(&want, scope) {
                continue;
            }
            let last = last.as_deref().and_then(Dt::parse);
            let expires = exp.as_deref().and_then(Dt::parse);
            if !gate_ok(
                state.as_deref().unwrap_or(""),
                fires.unwrap_or(0),
                maxf,
                cool.unwrap_or(0),
                last,
                expires,
                now,
            ) {
                continue;
            }
            if let Some(got) = self.fire_if_eligible(iid, &low, scope, now)? {
                fired.push(got);
            }
        }
        Ok(fired)
    }

    /// Re-check on the flat file and fire under a try-lock. `None` if not fired or locked.
    fn fire_if_eligible(
        &self,
        iid: &str,
        low: &str,
        have: &HashMap<String, String>,
        now: Dt,
    ) -> Result<Option<Intention>> {
        let Ok(_g) = self.lock(iid, Some(0.0)) else {
            return Ok(None);
        };
        let Some(cur) = self.read(iid)? else {
            return Ok(None);
        };
        if cur.kind != Kind::Event {
            return Ok(None);
        }
        let terms: Vec<String> = cur
            .trigger_terms
            .iter()
            .map(|t| py_strip(t).to_lowercase())
            .collect();
        if !(phrase_hit(&terms, low)
            && scope_ok(&cur.scope, have)
            && gate_ok(
                cur.state.as_str(),
                cur.fires,
                cur.max_fires,
                cur.cooldown_s,
                cur.last_fired_at,
                cur.expires_at,
                now,
            ))
        {
            return Ok(None);
        }
        let new = transition(&cur, "fire", Some(now), None)?; // transition() owns the state change
        let body = new.dump_indent2();
        atomic_write_bytes(&self.path(iid)?, body.as_bytes())?; // flat is truth: write it first
        if let Err(e) = self.index(&new, &body) {
            // flat is saved; reindex heals the derived row
            warn(&format!(
                "intention {iid} fired but index update failed: {e}"
            ));
        }
        Ok(Some(new))
    }

    // -- supersession sweep ---------------------------------------------

    /// `valid_to` of the origin memory, read defensively from the raw flat
    /// JSON: top-level `valid_to` first, then `metadata`.
    fn origin_valid_to(&self, memory_id: &str) -> Option<Dt> {
        check_id(memory_id).ok()?;
        for layer in LAYERS {
            let p = self.base.join(layer).join(format!("{memory_id}.json"));
            if !p.exists() {
                continue;
            }
            let raw: Value = serde_json::from_slice(&fs::read(&p).ok()?).ok()?;
            let top = raw.get("valid_to").filter(|v| is_truthy(v));
            let v = top.or_else(|| raw.get("metadata").and_then(|m| m.get("valid_to")));
            return v.and_then(Value::as_str).and_then(parse_dt_lenient);
        }
        None
    }

    /// Invalidate live intentions whose origin memory's `valid_to` has passed.
    /// Returns the ids changed. A future `valid_to`, a missing origin file or
    /// an unparseable value leaves the intention alone.
    pub fn sweep_supersession(&self, now: Option<Dt>) -> Result<Vec<String>> {
        let now = now.unwrap_or_else(Dt::now);
        let mut changed = Vec::new();
        let ids: Vec<String> = self.list(None)?.into_iter().map(|i| i.id).collect();
        for iid in ids {
            let Some(cur) = self.read(&iid)? else {
                continue;
            };
            let Some(origin) = cur.origin_memory_id.as_deref().filter(|o| !o.is_empty()) else {
                continue;
            };
            if cur.state.is_terminal() {
                continue;
            }
            if self.origin_valid_to(origin).is_some_and(|vt| vt <= now) {
                self.apply(&iid, "invalidate", Some(now), None)?;
                changed.push(iid);
            }
        }
        Ok(changed)
    }
}

fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `_parse_dt`: `Z` accepted, junk is `None`.
pub(crate) fn parse_dt_lenient(s: &str) -> Option<Dt> {
    if s.is_empty() {
        return None;
    }
    Dt::parse(s)
}

/// The text the prefilter scans: the whole prompt, or its head and tail
/// (`limit` chars in all) joined by NUL (which cannot join two words into a
/// phrase). A token split by the cut is dropped from the stub.
pub fn scan_text(text: &str, limit: i64) -> String {
    // Bytes bound chars: a short-enough text needs no count at all.
    if (text.len() as i64) <= limit {
        return text.to_string();
    }
    let total = text.chars().count();
    if (total as i64) <= limit {
        return text.to_string();
    }
    let half = ((limit / 2).max(1)) as usize;
    let head_end = text.char_indices().nth(half).map_or(text.len(), |(i, _)| i);
    let tail_start = text
        .char_indices()
        .rev()
        .nth(half - 1)
        .map_or(0, |(i, _)| i);
    let mut head = &text[..head_end];
    let mut tail = &text[tail_start..];
    let before = |end: usize| text[..end].chars().next_back();
    // The cut split a token: drop its stub.
    if before(head_end).is_some_and(is_word) && text[head_end..].chars().next().is_some_and(is_word)
    {
        head = head.trim_end_matches(is_word);
    }
    if before(tail_start).is_some_and(is_word) && tail.chars().next().is_some_and(is_word) {
        tail = tail.trim_start_matches(is_word);
    }
    format!("{head}\0{tail}")
}

fn prefilter_vals(i: &Intention) -> Vec<SqlValue> {
    let terms: Vec<String> = i
        .trigger_terms
        .iter()
        .filter(|t| !py_strip(t).is_empty())
        .map(|t| py_strip(&t.replace(SEP, " ")).to_lowercase())
        .collect();
    let mut sorted: Vec<&(String, String)> = i.scope.iter().collect();
    sorted.sort();
    let mut sc = Map::new();
    for (k, v) in sorted {
        sc.insert(k.clone(), v.clone().into());
    }
    vec![
        SqlValue::Text(terms.join(&SEP.to_string())),
        SqlValue::Text(crate::pyjson::dumps_compact(&Value::Object(sc))),
        SqlValue::Integer(i.cooldown_s),
        i.max_fires.map_or(SqlValue::Null, SqlValue::Integer),
        SqlValue::Integer(i.fires),
        i.last_fired_at
            .map_or(SqlValue::Null, |d| SqlValue::Text(d.isoformat())),
    ]
}

fn row_values(i: &Intention, body: &str) -> Vec<SqlValue> {
    let iso = |d: &Option<Dt>| d.map_or(SqlValue::Null, |x| SqlValue::Text(x.isoformat()));
    let mut v = vec![
        SqlValue::Text(i.id.clone()),
        SqlValue::Text(i.kind.as_str().into()),
        SqlValue::Text(i.state.as_str().into()),
        iso(&i.due_at),
        iso(&i.snooze_until),
        iso(&i.expires_at),
        i.origin_memory_id
            .clone()
            .map_or(SqlValue::Null, SqlValue::Text),
        SqlValue::Text(i.channel.clone()),
        SqlValue::Text(body.to_string()),
    ];
    v.extend(prefilter_vals(i));
    v
}

/// Hook entry point: [`IntentionStore::prefilter`] that can never fail. Any
/// failure (missing dir, locked or corrupt database, bad input) returns an
/// empty list. SQLite waits at most 50 ms, intention locks are try-locks and
/// the whole call shares one deadline.
pub fn prefilter_safe(
    base: impl AsRef<Path>,
    text: &str,
    scope: &HashMap<String, String>,
    now: Option<Dt>,
) -> Vec<Intention> {
    let t0 = Instant::now();
    let base = base.as_ref().to_path_buf();
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // A declared reader (SKMEMORY_READER ceiling) gets nothing: Python refuses it
        // every intention tool and its inject hook does nothing.
        let owner = OwnerScope::from_process().ok()?;
        let store = IntentionStore::open_with(&base, 50, None, &owner).ok()?;
        store
            .prefilter(
                text,
                scope,
                now,
                &PrefilterOpts {
                    t0: Some(t0),
                    ..Default::default()
                },
            )
            .ok()
    }));
    run.ok().flatten().unwrap_or_default()
}
