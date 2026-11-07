//! The per-class retrieval bench (`skmemory/eval/class_bench.py`) over the
//! vendored synthetic v1 fixtures in `tests/fixtures/eval-v1/`.
//!
//! [`run_bench`] loads `corpus-v1.jsonl` and `queries-v1.jsonl`, indexes the
//! corpus into a throwaway SQLite index, and scores five strategies per query
//! class: the baselines `bm25_only`, `grep_substring`, `vector_only`,
//! `fallback_chain`, the oracle `full_context`, and the `candidate`, which runs
//! this crate's own pipeline: the same arms Python's `MemoryStore.search` runs
//! (hashed vector, SQLite `LIKE`, FTS5 bm25, temporal), then [`crate::search`].
//! [`format_table`] prints the table Python prints.
//!
//! The dense signal is the bench's hashed bag-of-concepts embedding
//! (blake2b-8 buckets, 512 dims), not a semantic model: the `semantic` class
//! does not measure semantic retrieval offline, exactly as in Python.
//!
//! What is deliberately not ported (all comparison-only or inert here):
//! * the `reimpl_fusion` column (`fusion_bench`'s offline reimplementation, not
//!   gated in Python either);
//! * the full `parse_time_window` (see [`time_window`]: a subset that is
//!   checked against Python on every fixture query);
//! * the graph arm, which `SKMEMORY_GRAPH_ARM` leaves OFF by default, so the
//!   Python candidate never runs it either.
//!
//! Parity: Python's `_to_memory` passes the fixture `kind`, `outcome` and
//! `event_at` to the `Memory` (SKMEM-P1-12), so procedure ordering and outcome
//! handling see the real fixture values here too.

use crate::audience::OwnerScope;
use crate::embed::{EmbedError, Embedder, VECTOR_DIM};
use crate::engine::{Arms, Attach, Engine, EngineOptions, Filters, MemoryArm, SearchRequest};
use crate::env::{casefold, Env};
use crate::fusion::{self, ArmInput, ArmOutcome, DEFAULT_ARM_CAP};
use crate::index::{Index, SearchFilters};
use crate::model::{Layer, Memory};
use crate::pytime::Dt;
use crate::search::{search, Candidate, SearchInput};
use crate::validity::{normalize_as_of, valid_at};
use rusqlite::params_from_iter;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Recall cut-off.
pub const K: usize = 5;
/// The bench's own abstain threshold (cosine of top-1). Not fit to the fixtures.
pub const DEFAULT_ABSTAIN: f64 = 0.30;
/// The fixture clock the temporal arm and decay are evaluated at.
pub const FIXTURE_NOW: &str = "2026-11-01T12:00:00+00:00";
const EMBED_DIM: usize = 512;
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const EXPECTED_FAIL_REASON: &str = "still below gate with as_of applied (SKMEM-P3-03)";
const CANDIDATE: &str = "candidate";
const ORACLE: &str = "full_context";
/// Strategies with a real ranking, in table order (`reimpl_fusion` is not ported).
pub const SCORED: [&str; 5] = [
    "bm25_only",
    "grep_substring",
    "vector_only",
    "fallback_chain",
    CANDIDATE,
];
/// Every strategy, in table order.
pub const ALL_STRATEGIES: [&str; 6] = [
    "bm25_only",
    "grep_substring",
    "vector_only",
    "fallback_chain",
    CANDIDATE,
    ORACLE,
];

// ---------------------------------------------------------------------------
// Tokens, concepts, the hashed embedding
// ---------------------------------------------------------------------------

/// `fusion_bench._tokenize`: casefold, `[a-z0-9]+` runs of two or more chars.
pub fn tokenize(text: &str) -> Vec<String> {
    let folded = casefold(text);
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in folded.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            cur.push(c);
        } else {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            }
            cur.clear();
        }
    }
    out
}

/// `fusion_bench._CONCEPT_SYNONYMS`: surface form to concept key.
fn synonym(tok: &str) -> &str {
    match tok {
        "postgres" | "postgresql" | "pg" | "database" => "pg",
        "sqlite" | "index" | "cache" => "localindex",
        "embedding" | "embeddings" | "vector" | "vectors" | "semantic" | "cosine"
        | "similarity" | "nearest" => "vectorsearch",
        "keyword" | "bm25" | "lexical" | "fulltext" => "lexicalsearch",
        "crash" | "unclean" | "fsync" | "durable" | "durability" | "atomic" | "rename"
        | "pagecache" | "powerloss" | "suspend" => "durability",
        "orphan" | "orphans" | "missing" | "lost" | "vanished" | "gone" | "truncated" | "stub"
        | "preview" => "dataloss",
        "recovery" | "restore" | "restored" | "rescue" | "rebuild" | "reconstruct" => "restore",
        "syncthing" | "sync" | "replicate" | "replication" | "mirror" => "replication",
        "alert" | "alerts" | "monitor" | "monitoring" | "health" | "audit" | "drift" | "check" => {
            "monitoring"
        }
        "private" | "intimate" | "confidential" | "secret" | "redact" | "privacy" => "privacy",
        "gateway" | "inference" | "router" | "model" | "llm" => "inference",
        "gpu" | "arc" | "cuda" => "accelerator",
        "backup" | "backups" | "dump" | "snapshot" | "retention" | "tarball" => "backups",
        "cron" | "timer" | "systemd" | "schedule" | "nightly" | "weekly" => "scheduling",
        "queue" | "worker" | "job" | "task" => "worker",
        other => other,
    }
}

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// BLAKE2b with an 8 byte digest and no key, read as a big-endian integer
/// (`int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big")`).
pub fn blake2b8(data: &[u8]) -> u64 {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 8;
    let mut t: u128 = 0;
    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[][..]]
    } else {
        data.chunks(128).collect()
    };
    for (n, chunk) in chunks.iter().enumerate() {
        let last = n + 1 == chunks.len();
        t += chunk.len() as u128;
        let mut block = [0u8; 128];
        block[..chunk.len()].copy_from_slice(chunk);
        let m: Vec<u64> = block
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t as u64;
        v[13] ^= (t >> 64) as u64;
        if last {
            v[14] = !v[14];
        }
        for r in 0..12 {
            let s = &SIGMA[r % 10];
            let g = |v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64| {
                v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
                v[d] = (v[d] ^ v[a]).rotate_right(32);
                v[c] = v[c].wrapping_add(v[d]);
                v[b] = (v[b] ^ v[c]).rotate_right(24);
                v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
                v[d] = (v[d] ^ v[a]).rotate_right(16);
                v[c] = v[c].wrapping_add(v[d]);
                v[b] = (v[b] ^ v[c]).rotate_right(63);
            };
            g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }
    u64::from_be_bytes(h[0].to_le_bytes())
}

/// The hashed bucket a concept lands in.
pub fn concept_bucket(concept: &str) -> usize {
    (blake2b8(concept.as_bytes()) % EMBED_DIM as u64) as usize
}

/// `fusion_bench._concept_vector`: hashed bag of concepts, L2 normalised.
pub fn concept_vector(text: &str) -> Vec<f64> {
    let mut vec = vec![0.0f64; EMBED_DIM];
    for tok in tokenize(text) {
        vec[concept_bucket(synonym(&tok))] += 1.0;
    }
    let mut sum = 0.0;
    for v in &vec {
        sum += v * v;
    }
    let norm = sum.sqrt();
    if norm == 0.0 {
        return vec;
    }
    vec.into_iter().map(|v| v / norm).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// Unix seconds (UTC) of `dt`, the clock [`time_window`] reads.
pub(crate) fn unix_secs(dt: Dt) -> i64 {
    let (y, mo, d, h, mi, s, _) = dt.to_utc().fields();
    days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s
}

// ---------------------------------------------------------------------------
// Time window (subset of fusion.parse_time_window, strict mode)
// ---------------------------------------------------------------------------

pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `_utc_z`: `2026-06-30T23:59:59.999999Z`, fraction only when non-zero.
fn utc_z(secs: i64, us: i64) -> String {
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    let frac = if us == 0 {
        String::new()
    } else {
        format!(".{us:06}")
    };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{frac}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Half-open day-number span to the inclusive `(after, before)` pair.
fn span(start_day: i64, end_day_exclusive: i64) -> (String, String) {
    (
        utc_z(start_day * 86_400, 0),
        utc_z(end_day_exclusive * 86_400 - 1, 999_999),
    )
}

fn shift_month(year: i64, month: i64, delta: i64) -> (i64, i64) {
    let idx = year * 12 + (month - 1) + delta;
    (idx.div_euclid(12), idx.rem_euclid(12) + 1)
}

fn month_bounds(year: i64, month: i64) -> (String, String) {
    let (ny, nm) = shift_month(year, month, 1);
    span(days_from_civil(year, month, 1), days_from_civil(ny, nm, 1))
}

fn week_span(day: i64) -> (String, String) {
    // 1970-01-01 (day 0) was a Thursday; Monday is weekday 0.
    let monday = day - (day + 3).rem_euclid(7);
    span(monday, monday + 7)
}

fn num_word(w: &str) -> Option<i64> {
    const WORDS: [&str; 12] = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve",
    ];
    if w == "a" || w == "an" {
        return Some(1);
    }
    if !w.is_empty() && w.len() <= 3 && w.bytes().all(|b| b.is_ascii_digit()) {
        return w.parse().ok();
    }
    WORDS.iter().position(|x| *x == w).map(|i| i as i64 + 1)
}

fn unit_of(w: &str) -> Option<&'static str> {
    match w {
        "day" | "days" => Some("day"),
        "week" | "weeks" => Some("week"),
        "month" | "months" => Some("month"),
        _ => None,
    }
}

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const WEEKDAYS: [&str; 7] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];
const DATE_CUES: [&str; 10] = [
    "in", "during", "since", "until", "till", "before", "after", "from", "on", "by",
];

/// The window the temporal arm derives from `query`, as `(created_after,
/// created_before)` UTC strings, or `None` when the arm abstains.
///
/// A SUBSET of Python's `parse_time_window(strict=True)`, enough for the v1
/// fixtures and checked against Python on every fixture query
/// (`tests/class_bench.rs::time_windows_match_python`): `yesterday`, `today`,
/// `N day|week|month(s) ago`, `last|past N day|week|month(s)`, `last <weekday>`,
/// `this|last week|month`, and an unambiguous month name directly after a
/// time preposition (`in june`). Not handled, so they abstain here where
/// Python would answer: ISO dates and months, bare years, `may` and `march`,
/// month plus day, and `the past month`.
pub fn time_window(query: &str, now_secs: i64) -> Option<(String, String)> {
    let text = casefold(query);
    let words: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let today = now_secs.div_euclid(86_400);
    let (ty, tm, _) = civil_from_days(today);
    let at = |i: usize| words.get(i).copied();
    if words.contains(&"yesterday") {
        return Some(span(today - 1, today));
    }
    if words.contains(&"today") {
        return Some(span(today, today + 1));
    }
    for (i, &word) in words.iter().enumerate() {
        if let (Some(n), Some(u), Some("ago")) =
            (num_word(word), at(i + 1).and_then(unit_of), at(i + 2))
        {
            return Some(match u {
                "day" => span(today - n, today - n + 1),
                "week" => week_span(today - 7 * n),
                _ => {
                    let (y, m) = shift_month(ty, tm, -n);
                    month_bounds(y, m)
                }
            });
        }
    }
    for (i, &word) in words.iter().enumerate() {
        if matches!(word, "last" | "past") {
            if let (Some(n), Some(u)) = (at(i + 1).and_then(num_word), at(i + 2).and_then(unit_of))
            {
                let start = if u == "month" {
                    let (y, m) = shift_month(ty, tm, -n);
                    let (_, _, d) = civil_from_days(today);
                    let day = if d <= days_in_month(y, m) {
                        days_from_civil(y, m, d)
                    } else {
                        let (ny, nm) = shift_month(y, m, 1);
                        days_from_civil(ny, nm, 1) - 1
                    };
                    day * 86_400 + now_secs.rem_euclid(86_400)
                } else {
                    now_secs - n * if u == "week" { 7 } else { 1 } * 86_400
                };
                return Some((utc_z(start, 0), utc_z(now_secs, 0)));
            }
        }
    }
    for (i, &word) in words.iter().enumerate() {
        if word == "last" {
            if let Some(wd) = at(i + 1).and_then(|w| WEEKDAYS.iter().position(|x| *x == w)) {
                let today_wd = (today + 3).rem_euclid(7);
                let mut back = (today_wd - wd as i64).rem_euclid(7);
                if back == 0 {
                    back = 7;
                }
                return Some(span(today - back, today - back + 1));
            }
        }
    }
    for (i, &word) in words.iter().enumerate() {
        let next = at(i + 1);
        match (word, next) {
            ("this", Some("week")) if at(i + 2) != Some("of") => return Some(week_span(today)),
            ("last", Some("week")) if at(i + 2) != Some("of") => return Some(week_span(today - 7)),
            ("this", Some("month")) => return Some(month_bounds(ty, tm)),
            ("last", Some("month")) => {
                let (y, m) = shift_month(ty, tm, -1);
                return Some(month_bounds(y, m));
            }
            _ => {}
        }
    }
    for (i, &word) in words.iter().enumerate() {
        if !DATE_CUES.contains(&word) {
            continue;
        }
        let j = if at(i + 1) == Some("the") {
            i + 2
        } else {
            i + 1
        };
        if let Some(m) = at(j).and_then(|w| MONTHS.iter().position(|x| *x == w)) {
            let month = m as i64 + 1;
            if month == 3 || month == 5 {
                continue;
            }
            let year = if month <= tm { ty } else { ty - 1 };
            return Some(month_bounds(year, month));
        }
    }
    None
}

fn days_in_month(y: i64, m: i64) -> i64 {
    let (ny, nm) = shift_month(y, m, 1);
    days_from_civil(ny, nm, 1) - days_from_civil(y, m, 1)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One corpus memory with the derived fields every strategy reads.
pub struct Doc {
    pub memory: Memory,
    /// `title content tags`, what bm25 and the hashed embedding read.
    text: String,
    cvec: Vec<f64>,
    /// `json.dumps` of the fixture row, casefolded (what `grep_substring` greps).
    flat: String,
}

/// One bench query.
#[derive(Clone, Debug)]
pub struct Query {
    pub id: String,
    pub class: String,
    pub query: String,
    pub gold: Vec<String>,
    pub must_not: Vec<String>,
    pub as_of: Option<String>,
    pub reader: String,
    pub gate: f64,
}

fn jstr(v: &Value, k: &str) -> Result<String, String> {
    v.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing string field {k}"))
}

fn jopt(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

fn jlist(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(n, l)| {
            serde_json::from_str(l).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))
        })
        .collect()
}

/// `model_dump()` of the fixture's `CorpusMemory`: its field order, `outcome` always present.
fn flat_dump(row: &Value) -> String {
    let mut m = serde_json::Map::new();
    for k in [
        "id",
        "title",
        "content",
        "layer",
        "tags",
        "created_at",
        "event_at",
        "kind",
        "source_tier",
        "emotional",
        "context_tag",
        "valid_from",
        "valid_to",
        "supersedes_id",
        "schema_version",
        "outcome",
    ] {
        m.insert(k.to_string(), row.get(k).cloned().unwrap_or(Value::Null));
    }
    casefold(&crate::pyjson::dumps_compact(&Value::Object(m)))
}

fn load_corpus(dir: &Path) -> Result<Vec<Doc>, String> {
    read_jsonl(&dir.join("corpus-v1.jsonl"))?
        .iter()
        .map(|r| {
            let tags = jlist(r, "tags");
            let mut m = Memory::new(&jstr(r, "title")?, &jstr(r, "content")?);
            m.id = jstr(r, "id")?;
            m.layer = Layer::parse(&jstr(r, "layer")?).ok_or("bad layer")?;
            m.tags = tags.clone();
            m.created_at = jstr(r, "created_at")?;
            m.updated_at = m.created_at.clone();
            m.context_tag = jstr(r, "context_tag")?;
            m.event_at = jopt(r, "event_at");
            m.valid_from = jopt(r, "valid_from");
            m.valid_to = jopt(r, "valid_to");
            m.supersedes_id = jopt(r, "supersedes_id");
            m.source_tier = jstr(r, "source_tier")?;
            m.schema_version = r.get("schema_version").and_then(Value::as_i64).unwrap_or(2);
            m.kind = jstr(r, "kind")?;
            m.outcome = jopt(r, "outcome");
            let e = r.get("emotional").ok_or("missing emotional")?;
            m.emotional.intensity = e.get("intensity").and_then(Value::as_f64).unwrap_or(0.0);
            m.emotional.valence = e.get("valence").and_then(Value::as_f64).unwrap_or(0.0);
            if !matches!(m.context_tag.as_str(), "@public" | "@chef-only") {
                return Err(format!("unsupported context_tag {}", m.context_tag));
            }
            let text = format!("{} {} {}", m.title, m.content, tags.join(" "));
            Ok(Doc {
                cvec: concept_vector(&text),
                flat: flat_dump(r),
                text,
                memory: m,
            })
        })
        .collect()
}

fn load_queries(dir: &Path) -> Result<Vec<Query>, String> {
    read_jsonl(&dir.join("queries-v1.jsonl"))?
        .iter()
        .map(|r| {
            Ok(Query {
                id: jstr(r, "id")?,
                class: jstr(r, "class")?,
                query: jstr(r, "query")?,
                gold: jlist(r, "gold_ids"),
                must_not: jlist(r, "must_not_ids"),
                as_of: jopt(r, "as_of"),
                reader: jstr(r, "reader")?,
                gate: r
                    .get("gate_recall_at_5")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

fn arm_semantic(query: &str, docs: &[&Doc], cap: usize) -> Vec<String> {
    let qv = concept_vector(query);
    let mut scored: Vec<(&str, f64)> = docs
        .iter()
        .map(|d| (d.memory.id.as_str(), cosine(&qv, &d.cvec)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    scored
        .into_iter()
        .take(cap)
        .map(|(i, _)| i.to_string())
        .collect()
}

fn arm_bm25(query: &str, docs: &[&Doc], cap: usize) -> Vec<String> {
    let toks: Vec<Vec<String>> = docs.iter().map(|d| tokenize(&d.text)).collect();
    let n = docs.len() as f64;
    let avgdl = if docs.is_empty() {
        0.0
    } else {
        toks.iter().map(Vec::len).sum::<usize>() as f64 / n
    };
    let mut df: HashMap<&str, i64> = HashMap::new();
    for t in &toks {
        for term in t.iter().map(String::as_str).collect::<BTreeSet<_>>() {
            *df.entry(term).or_insert(0) += 1;
        }
    }
    let q_terms = tokenize(query);
    let mut scored: Vec<(&str, f64)> = Vec::new();
    for (d, t) in docs.iter().zip(&toks) {
        let mut tf: HashMap<&str, i64> = HashMap::new();
        for x in t {
            *tf.entry(x.as_str()).or_insert(0) += 1;
        }
        let dl = t.len() as f64;
        let mut score = 0.0;
        for term in &q_terms {
            let f = tf.get(term.as_str()).copied().unwrap_or(0);
            if f == 0 {
                continue;
            }
            let dft = df[term.as_str()] as f64;
            let idf = (1.0 + (n - dft + 0.5) / (dft + 0.5)).ln();
            let ratio = if avgdl != 0.0 { dl / avgdl } else { 0.0 };
            let denom = f as f64 + BM25_K1 * (1.0 - BM25_B + BM25_B * ratio);
            score += idf * (f as f64 * (BM25_K1 + 1.0)) / denom;
        }
        if score > 0.0 {
            scored.push((d.memory.id.as_str(), score));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    scored
        .into_iter()
        .take(cap)
        .map(|(i, _)| i.to_string())
        .collect()
}

fn strat_grep(query: &str, docs: &[&Doc], k: usize) -> Vec<String> {
    let tokens = tokenize(query);
    let mut scored: Vec<(&str, usize, usize)> = Vec::new();
    for d in docs {
        let counts: Vec<usize> = tokens
            .iter()
            .map(|t| d.flat.matches(t.as_str()).count())
            .collect();
        let total: usize = counts.iter().sum();
        let distinct = counts.iter().filter(|c| **c > 0).count();
        if total > 0 {
            scored.push((d.memory.id.as_str(), total, distinct));
        }
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(b.0)));
    scored
        .into_iter()
        .take(k)
        .map(|(i, _, _)| i.to_string())
        .collect()
}

fn strat_fallback(query: &str, docs: &[&Doc], k: usize) -> Vec<String> {
    let primary = arm_semantic(query, docs, k);
    if primary.is_empty() {
        arm_bm25(query, docs, k)
    } else {
        primary
    }
}

fn oracle(q: &Query, docs: &[&Doc]) -> Vec<String> {
    let ids: BTreeSet<&str> = docs.iter().map(|d| d.memory.id.as_str()).collect();
    let gold: Vec<&String> = q.gold.iter().filter(|i| ids.contains(i.as_str())).collect();
    let mut out: Vec<String> = gold.iter().map(|s| s.to_string()).collect();
    for d in docs {
        if !gold.iter().any(|g| **g == d.memory.id) {
            out.push(d.memory.id.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The candidate: the Rust pipeline over a real SQLite index
// ---------------------------------------------------------------------------

struct Store<'a> {
    docs: &'a [Doc],
    by_id: HashMap<&'a str, &'a Doc>,
    index: Index,
    /// The bench is an owner run over synthetic fixtures: it says so explicitly.
    owner: OwnerScope,
    found: HashMap<String, Candidate>,
    now_secs: i64,
}

impl<'a> Store<'a> {
    fn passes(&self, d: &Doc, ctx: Option<&str>, as_of: Option<&str>) -> bool {
        let m = &d.memory;
        if let Some(t) = as_of {
            if !valid_at(
                m.valid_from.as_deref(),
                Some(&m.created_at),
                m.valid_to.as_deref(),
                t,
            ) {
                return false;
            }
        }
        ctx.is_none_or(|c| m.context_tag == c)
    }

    fn semantic(
        &self,
        q: &str,
        limit: usize,
        ctx: Option<&str>,
        as_of: Option<&str>,
    ) -> Vec<String> {
        let qv = concept_vector(q);
        let mut scored: Vec<(&str, f64)> = self
            .docs
            .iter()
            .filter(|d| self.passes(d, ctx, as_of))
            .map(|d| (d.memory.id.as_str(), cosine(&qv, &d.cvec)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
        scored
            .into_iter()
            .take(limit)
            .map(|(i, _)| i.to_string())
            .collect()
    }

    /// `SQLiteBackend.search_text`: AND of per-word `LIKE`, then OR ranked by hits.
    fn like(&self, q: &str, limit: usize) -> Vec<String> {
        let words: Vec<&str> = q.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let clause = "(title LIKE ? OR summary LIKE ? OR content_preview LIKE ? OR tags LIKE ?)";
        let pats = |ws: &[&str]| -> Vec<String> {
            ws.iter().flat_map(|w| vec![format!("%{w}%"); 4]).collect()
        };
        let run = |sql: String, mut args: Vec<String>| -> Vec<String> {
            args.push(limit.to_string());
            let mut st = match self.index.connection(&self.owner).prepare(&sql) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            st.query_map(params_from_iter(args), |r| r.get::<_, String>(0))
                .map(|rows| rows.filter_map(Result::ok).collect())
                .unwrap_or_default()
        };
        let and = vec![clause; words.len()].join(" AND ");
        let mut ids = run(
            format!("SELECT id FROM memories WHERE {and} ORDER BY created_at DESC LIMIT ?"),
            pats(&words),
        );
        if ids.is_empty() && words.len() > 1 {
            let or = vec![clause; words.len()].join(" OR ");
            let score =
                vec![format!("CASE WHEN {clause} THEN 1 ELSE 0 END"); words.len()].join(" + ");
            let mut args = pats(&words);
            args.extend(pats(&words));
            ids = run(
                format!("SELECT id FROM memories WHERE {or} ORDER BY ({score}) DESC, created_at DESC LIMIT ?"),
                args,
            );
        }
        ids
    }

    fn term_df(&self, token: &str) -> i64 {
        let cleaned = token.replace('"', " ");
        let cleaned = crate::model::py_strip(&cleaned);
        if cleaned.is_empty() {
            return 0;
        }
        self.index
            .connection(&self.owner)
            .query_row(
                "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?",
                [format!("\"{cleaned}\"")],
                |r| r.get(0),
            )
            .unwrap_or(0)
    }

    /// `list_memories(created_after, created_before, as_of)`, newest first.
    fn created_between(
        &self,
        after: &str,
        before: &str,
        as_of: Option<&str>,
        limit: usize,
    ) -> Vec<String> {
        let norm = |v: &str| {
            normalize_as_of(v)
                .ok()
                .flatten()
                .unwrap_or_else(|| v.to_string())
        };
        let mut sql = String::from(
            "SELECT id FROM memories WHERE replace(created_at, 'Z', '+00:00') >= ? \
             AND replace(created_at, 'Z', '+00:00') < ?",
        );
        let mut args = vec![norm(after), norm(before)];
        if let Some(a) = as_of {
            sql.push_str(
                " AND COALESCE(julianday(valid_from), julianday(created_at), 0) <= julianday(?) \
                 AND (julianday(valid_to) IS NULL OR julianday(valid_to) > julianday(?))",
            );
            args.push(a.to_string());
            args.push(a.to_string());
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ?");
        args.push(limit.to_string());
        let Ok(mut st) = self.index.connection(&self.owner).prepare(&sql) else {
            return Vec::new();
        };
        st.query_map(params_from_iter(args), |r| r.get::<_, String>(0))
            .map(|rows| rows.filter_map(Result::ok).collect())
            .unwrap_or_default()
    }

    /// `MemoryStore.search` with fusion and rerank on, decay off.
    fn search(
        &self,
        query: &str,
        limit: usize,
        ctx: Option<&str>,
        as_of: Option<&str>,
    ) -> Vec<String> {
        let as_of = as_of.and_then(|a| normalize_as_of(a).ok().flatten());
        let as_of = as_of.as_deref();
        let pool = limit.max(limit * 3);
        let arm_limit = (pool * 3).max(DEFAULT_ARM_CAP);
        let collect = |ids: Vec<String>, check_ctx: bool| -> Vec<String> {
            ids.into_iter()
                .filter(|i| {
                    self.by_id
                        .get(i.as_str())
                        .is_some_and(|d| self.passes(d, if check_ctx { ctx } else { None }, as_of))
                })
                .collect()
        };
        let semantic = collect(self.semantic(query, arm_limit, ctx, as_of), false);
        let lexical = collect(self.like(query, arm_limit), true);
        let ctx_tags: Option<Vec<String>> = ctx.map(|c| vec![c.to_string()]);
        let owner = self.owner.reader_scope();
        let local: Vec<String> = self
            .index
            .search_lexical(
                query,
                arm_limit,
                &SearchFilters {
                    layer: None,
                    tags: &[],
                    source: None,
                    context_tags: ctx_tags.as_deref(),
                    as_of,
                },
                owner,
            )
            .map(|rows| rows.into_iter().map(|r| r.id).collect())
            .unwrap_or_default();
        let lexical_local = collect(local, false);
        let temporal = match time_window(query, self.now_secs) {
            Some((a, b)) => collect(self.created_between(&a, &b, as_of, arm_limit), true),
            None => Vec::new(),
        };
        let cap = DEFAULT_ARM_CAP;
        let arms: Vec<ArmOutcome> = vec![
            fusion::run_arm("semantic", ArmInput::Ids(semantic), cap),
            fusion::run_arm("lexical", ArmInput::Ids(lexical), cap),
            fusion::run_arm("lexical_local", ArmInput::Ids(lexical_local), cap),
            fusion::run_arm(
                "graph",
                ArmInput::Disabled(
                    "disabled by config (SKMEMORY_GRAPH_ARM is off, the default)".into(),
                ),
                cap,
            ),
            fusion::run_arm("temporal", ArmInput::Ids(temporal), cap),
        ];
        let df = |t: &str| self.term_df(t);
        let input = SearchInput {
            query,
            scope: owner,
            limit,
            arms: &arms,
            found: &self.found,
            has_pg_bm25: false,
            df: Some(&df),
            usage: None,
            decay_as_of: Dt::parse(FIXTURE_NOW).expect("fixture clock"),
        };
        match search(&input, &Env::empty()) {
            Some(t) => t.page,
            // No arm answered: Python falls to the vector backend's text search.
            None => self
                .semantic(query, pool, None, None)
                .into_iter()
                .filter(|i| {
                    self.by_id
                        .get(i.as_str())
                        .is_some_and(|d| self.passes(d, ctx, as_of))
                })
                .take(limit)
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// The candidate through the Engine (SKMEM-R-07C)
// ---------------------------------------------------------------------------

/// `title content tags`, the text the bench embeds for a memory.
pub fn doc_text(m: &Memory) -> String {
    format!("{} {} {}", m.title, m.content, m.tags.join(" "))
}

/// The hashed bag-of-concepts vector, zero-padded from 512 to the 1024 dims of
/// the pgvector column. Zero padding changes no dot product and no norm, so a
/// cosine over the padded vectors equals the bench's cosine.
pub fn hashed_dense(text: &str) -> Vec<f32> {
    let mut v: Vec<f32> = concept_vector(text).into_iter().map(|x| x as f32).collect();
    v.resize(VECTOR_DIM, 0.0);
    v
}

/// An [`Embedder`] that produces [`hashed_dense`] (the bench's offline "model").
pub struct HashedEmbedder;

impl Embedder for HashedEmbedder {
    fn embed(&self, text: &str, _required: bool) -> Result<Vec<f32>, EmbedError> {
        Ok(hashed_dense(text))
    }
}

/// The bench's hashed `semantic` arm as an in-memory [`MemoryArm`]: cosine
/// above zero, ties by id, with the filters the pgvector arm pushes down
/// (`context_tag` and `as_of`).
pub struct HashedSemantic {
    docs: Vec<(Memory, Vec<f64>)>,
}

impl HashedSemantic {
    pub fn new(memories: &[Memory]) -> HashedSemantic {
        HashedSemantic {
            docs: memories
                .iter()
                .map(|m| (m.clone(), concept_vector(&doc_text(m))))
                .collect(),
        }
    }
}

impl MemoryArm for HashedSemantic {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        f: &Filters,
        _scope: &crate::audience::ReaderScope,
    ) -> Result<Vec<Memory>, String> {
        let qv = concept_vector(query);
        let mut scored: Vec<(&Memory, f64)> = self
            .docs
            .iter()
            .filter(|(m, _)| {
                f.as_of.as_deref().is_none_or(|t| {
                    valid_at(
                        m.valid_from.as_deref(),
                        Some(&m.created_at),
                        m.valid_to.as_deref(),
                        t,
                    )
                }) && f
                    .context_tags
                    .as_ref()
                    .is_none_or(|c| c.contains(&m.context_tag))
            })
            .map(|(m, v)| (m, cosine(&qv, v)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.id.cmp(&b.0.id)));
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(m, _)| m.clone())
            .collect())
    }
}

/// Which arms the Engine candidate runs with.
pub enum EngineArms {
    /// The in-memory hashed `semantic` arm; `lexical` is SQLite `LIKE`. This is
    /// the configuration Python's bench runs, so its pages are the golden ones.
    Hashed,
    /// Arms built from the corpus (for example Postgres arms over a seeded
    /// throwaway database).
    Custom(Box<dyn Fn(&[Memory]) -> Result<Arms, String>>),
}

/// [`run_bench`] with the candidate run through [`Engine::search`].
pub fn run_bench_engine(fixture_dir: &Path, arms: EngineArms) -> Result<BenchResult, String> {
    run_bench_inner(fixture_dir, Some(arms))
}

struct EngineCandidate {
    engine: Engine,
    owner: OwnerScope,
    now: Dt,
    _tmp: TempDir,
}

impl EngineCandidate {
    fn new(docs: &[Doc], arms: EngineArms, now: Dt) -> Result<EngineCandidate, String> {
        let tmp = TempDir::new()?;
        let memories: Vec<Memory> = docs.iter().map(|d| d.memory.clone()).collect();
        let arms = match arms {
            EngineArms::Hashed => Arms {
                semantic: Attach::ready(Box::new(HashedSemantic::new(&memories))),
                lexical: Attach::Absent,
                graph: Attach::Disabled(
                    "disabled by config (SKMEMORY_GRAPH_ARM is off, the default)".into(),
                ),
            },
            EngineArms::Custom(f) => f(&memories)?,
        };
        let owner = OwnerScope::from_process().map_err(|e| e.to_string())?;
        let engine = Engine::open(
            &tmp.0,
            EngineOptions {
                audience_config: None,
                fts: true,
                read_only: Some(false),
                arms,
            },
        )
        .map_err(|e| e.to_string())?;
        for m in &memories {
            engine.save(m, &owner).map_err(|e| e.to_string())?;
        }
        Ok(EngineCandidate {
            engine,
            owner,
            now,
            _tmp: tmp,
        })
    }

    fn search(&self, q: &str, limit: usize, ctx: Option<&str>, as_of: Option<&str>) -> Vec<String> {
        let as_of = as_of.and_then(|a| normalize_as_of(a).ok().flatten());
        let ctx_tags: Option<Vec<String>> = ctx.map(|c| vec![c.to_string()]);
        let out = self.engine.search(
            &SearchRequest {
                query: q,
                limit,
                layer: None,
                tags: &[],
                source: None,
                context_tags: ctx_tags.as_deref(),
                as_of: as_of.as_deref(),
                now: Some(self.now),
            },
            self.owner.reader_scope(),
            &Env::empty(),
        );
        out.map(|o| o.memories.into_iter().map(|m| m.id).collect())
            .unwrap_or_default()
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<TempDir, String> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p =
            std::env::temp_dir().join(format!("skmem-class-bench-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).map_err(|e| e.to_string())?;
        Ok(TempDir(p))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

fn recall_at_k(got: &[String], gold: &[String], k: usize) -> f64 {
    let rel: BTreeSet<&str> = gold.iter().map(String::as_str).collect();
    if rel.is_empty() {
        return 0.0;
    }
    let top: BTreeSet<&str> = got.iter().take(k).map(String::as_str).collect();
    top.intersection(&rel).count() as f64 / rel.len() as f64
}

fn rr(got: &[String], gold: &[String]) -> f64 {
    got.iter()
        .position(|i| gold.contains(i))
        .map_or(0.0, |p| 1.0 / (p as f64 + 1.0))
}

fn mean(v: impl IntoIterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = v.into_iter().collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// One strategy's numbers for one class.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StratStats {
    pub recall_at_5: Option<f64>,
    pub mrr: Option<f64>,
    /// `absent` class only.
    pub abstain_correct: Option<f64>,
    /// Classes with gold only.
    pub false_abstain: Option<f64>,
    /// `privacy` class only.
    pub leak_count: Option<usize>,
}

/// One query's candidate page, for parity checks.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub class: String,
    pub candidate_page: Vec<String>,
}

/// A candidate query that missed its gate.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub id: String,
    pub class: String,
    pub metric: &'static str,
    pub value: f64,
    pub gate: f64,
    pub expected_fail: bool,
}

/// Everything [`run_bench`] measured.
#[derive(Clone, Debug)]
pub struct BenchResult {
    pub threshold: f64,
    pub corpus_tokens: usize,
    pub rows: Vec<Row>,
    /// Class name to (query count, strategy name to stats).
    pub by_class: BTreeMap<String, (usize, BTreeMap<&'static str, StratStats>)>,
    pub failures: Vec<Failure>,
    pub expected_fail_passing_today: Vec<String>,
    pub leak_count: usize,
    pub leak_count_unfiltered: usize,
}

/// Score every strategy on every query of the fixtures in `fixture_dir`
/// (`corpus-v1.jsonl`, `queries-v1.jsonl`). Offline and deterministic.
pub fn run_bench(fixture_dir: &Path) -> Result<BenchResult, String> {
    run_bench_inner(fixture_dir, None)
}

fn run_bench_inner(fixture_dir: &Path, engine: Option<EngineArms>) -> Result<BenchResult, String> {
    let docs = load_corpus(fixture_dir)?;
    let queries = load_queries(fixture_dir)?;
    let tmp = TempDir::new()?;
    // The bench is an owner run over synthetic fixtures; it refuses under a reader ceiling.
    let owner = OwnerScope::from_process().map_err(|e| e.to_string())?;
    let index = Index::open(&tmp.0, true).map_err(|e| e.to_string())?;
    for d in &docs {
        index
            .index_memory(&d.memory, Path::new(""), &owner)
            .map_err(|e| e.to_string())?;
    }
    let now = Dt::parse(FIXTURE_NOW).ok_or("fixture clock")?;
    let (y, mo, d, h, mi, s, _) = now.fields();
    let now_secs = days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s;
    let store = Store {
        by_id: docs.iter().map(|d| (d.memory.id.as_str(), d)).collect(),
        found: docs
            .iter()
            .map(|d| (d.memory.id.clone(), Candidate::from_memory(&d.memory)))
            .collect(),
        docs: &docs,
        index,
        owner,
        now_secs,
    };
    let cand = engine
        .map(|a| EngineCandidate::new(&docs, a, now))
        .transpose()?;
    let candidate_search =
        |q: &str, limit: usize, ctx: Option<&str>, as_of: Option<&str>| match &cand {
            Some(c) => c.search(q, limit, ctx, as_of),
            None => store.search(q, limit, ctx, as_of),
        };
    let all: Vec<&Doc> = docs.iter().collect();
    let public: Vec<&Doc> = docs
        .iter()
        .filter(|d| d.memory.context_tag == "@public")
        .collect();
    let thr = DEFAULT_ABSTAIN;
    let corpus_tokens: usize = docs.iter().map(|d| d.text.split_whitespace().count()).sum();

    struct Per {
        recall: Option<f64>,
        rr: Option<f64>,
        abstained: Option<bool>,
        must_not_hits: usize,
    }
    let mut rows = Vec::new();
    let mut per_q: Vec<(Query, bool, BTreeMap<&'static str, Per>, usize)> = Vec::new();
    for q in &queries {
        let visible: &[&Doc] = if q.reader == "@chef" { &all } else { &public };
        let has_gold = !q.gold.is_empty();
        let ctx = (q.reader != "@chef").then_some("@public");
        let mut per = BTreeMap::new();
        let mut page = Vec::new();
        for name in ALL_STRATEGIES {
            let got: Vec<String> = match name {
                ORACLE => oracle(q, visible),
                CANDIDATE => candidate_search(&q.query, 10, ctx, q.as_of.as_deref()),
                "bm25_only" => arm_bm25(&q.query, visible, 10),
                "grep_substring" => strat_grep(&q.query, visible, 10),
                "vector_only" => arm_semantic(&q.query, visible, 10),
                _ => strat_fallback(&q.query, visible, 10),
            };
            let sig = (name != ORACLE).then(|| {
                got.first().map_or(0.0, |top| {
                    cosine(&concept_vector(&q.query), &store.by_id[top.as_str()].cvec)
                })
            });
            per.insert(
                name,
                Per {
                    recall: has_gold.then(|| recall_at_k(&got, &q.gold, K)),
                    rr: has_gold.then(|| rr(&got, &q.gold)),
                    abstained: sig.map(|s| s < thr),
                    must_not_hits: got
                        .iter()
                        .take(10)
                        .filter(|i| q.must_not.contains(i))
                        .count(),
                },
            );
            if name == CANDIDATE {
                page = got;
            }
        }
        let mut unfiltered = 0;
        if q.class == "privacy" && q.reader != "@chef" {
            for f in [
                arm_bm25 as fn(&str, &[&Doc], usize) -> Vec<String>,
                strat_grep,
                arm_semantic,
                strat_fallback,
            ] {
                unfiltered += f(&q.query, &all, 10)
                    .iter()
                    .filter(|i| q.must_not.contains(i))
                    .count();
            }
            unfiltered += candidate_search(&q.query, 10, None, None)
                .iter()
                .filter(|i| q.must_not.contains(i))
                .count();
        }
        rows.push(Row {
            id: q.id.clone(),
            class: q.class.clone(),
            candidate_page: page,
        });
        let expected_fail = q.as_of.is_some() && (q.class == "temporal" || q.class == "conflict");
        per_q.push((q.clone(), expected_fail, per, unfiltered));
    }

    let classes: BTreeSet<&str> = queries.iter().map(|q| q.class.as_str()).collect();
    let mut by_class = BTreeMap::new();
    for c in classes {
        let crow: Vec<_> = per_q.iter().filter(|r| r.0.class == c).collect();
        let gold_rows: Vec<_> = crow.iter().filter(|r| !r.0.gold.is_empty()).collect();
        let mut strategies = BTreeMap::new();
        for name in ALL_STRATEGIES {
            let mut st = StratStats {
                recall_at_5: mean(gold_rows.iter().map(|r| r.2[name].recall.unwrap())),
                mrr: mean(gold_rows.iter().map(|r| r.2[name].rr.unwrap())),
                ..StratStats::default()
            };
            if name != ORACLE {
                if c == "absent" {
                    st.abstain_correct = mean(
                        crow.iter()
                            .map(|r| f64::from(u8::from(r.2[name].abstained == Some(true)))),
                    );
                } else if !gold_rows.is_empty() {
                    st.false_abstain = mean(
                        gold_rows
                            .iter()
                            .map(|r| f64::from(u8::from(r.2[name].abstained == Some(true)))),
                    );
                }
            }
            if c == "privacy" {
                st.leak_count = Some(crow.iter().map(|r| r.2[name].must_not_hits).sum());
            }
            strategies.insert(name, st);
        }
        by_class.insert(c.to_string(), (crow.len(), strategies));
    }

    let mut failures = Vec::new();
    let mut xpass = Vec::new();
    for (q, expected_fail, per, _) in &per_q {
        let cand = &per[CANDIDATE];
        let (val, gate, metric) = if q.class == "absent" {
            (
                f64::from(u8::from(cand.abstained == Some(true))),
                1.0,
                "abstain_correct",
            )
        } else if let Some(r) = cand.recall {
            (r, q.gate, "recall@5")
        } else {
            continue;
        };
        if val < gate - 1e-9 {
            failures.push(Failure {
                id: q.id.clone(),
                class: q.class.clone(),
                metric,
                value: val,
                gate,
                expected_fail: *expected_fail,
            });
        } else if *expected_fail && metric == "recall@5" {
            xpass.push(q.id.clone());
        }
    }
    let leak_count = by_class
        .get("privacy")
        .map(|(_, s)| s.values().filter_map(|x| x.leak_count).sum())
        .unwrap_or(0);
    Ok(BenchResult {
        threshold: thr,
        corpus_tokens,
        rows,
        by_class,
        failures,
        expected_fail_passing_today: xpass,
        leak_count,
        leak_count_unfiltered: per_q.iter().map(|r| r.3).sum(),
    })
}

fn f(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{x:.3}"))
}

/// The table Python's `class_bench.format_table` prints (without `reimpl_fusion`).
pub fn format_table(res: &BenchResult) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "skmemory per-class baselines (skmem, Rust)");
    let _ = writeln!(
        o,
        "- candidate = skmem::search over a real SQLite index (hashed vector, LIKE, FTS5 bm25, temporal arms); \
         reimpl_fusion is not ported (comparison only, not gated)"
    );
    let _ = writeln!(
        o,
        "- the offline vector arm is a HASHED BAG OF WORDS with a synonym map, NOT a semantic embedding: \
         the 'semantic' class is NOT a measure of semantic retrieval offline"
    );
    let _ = writeln!(
        o,
        "- graph arm: off by default (SKMEMORY_GRAPH_ARM), as in Python"
    );
    let _ = writeln!(
        o,
        "- abstain threshold {} (cosine of top-1) is BENCH-ONLY",
        res.threshold
    );
    let _ = writeln!(
        o,
        "- full_context cost {} tokens (whole corpus)\n",
        res.corpus_tokens
    );
    for metric in ["recall@5", "mrr"] {
        let hdr = format!(
            "{metric:<12}{}",
            ALL_STRATEGIES
                .iter()
                .map(|n| format!("{n:>16}"))
                .collect::<String>()
        );
        let _ = writeln!(o, "{hdr}\n{}", "-".repeat(hdr.len()));
        for (c, (_, st)) in &res.by_class {
            let cells: String = ALL_STRATEGIES
                .iter()
                .map(|n| {
                    let v = if metric == "mrr" {
                        st[n].mrr
                    } else {
                        st[n].recall_at_5
                    };
                    format!("{:>16}", f(v))
                })
                .collect();
            let _ = writeln!(o, "{c:<12}{cells}");
        }
        let _ = writeln!(o);
    }
    let hdr = format!(
        "{:<12}{}",
        "abstain",
        SCORED
            .iter()
            .map(|n| format!("{n:>16}"))
            .collect::<String>()
    );
    let _ = writeln!(o, "{hdr}\n{}", "-".repeat(hdr.len()));
    if let Some((_, a)) = res.by_class.get("absent") {
        let cells: String = SCORED
            .iter()
            .map(|n| format!("{:>16}", f(a[n].abstain_correct)))
            .collect();
        let _ = writeln!(o, "{:<12}{cells}", "absent ok");
    }
    let cells: String = SCORED
        .iter()
        .map(|n| {
            let m = mean(
                res.by_class
                    .values()
                    .filter_map(|(_, s)| s[n].false_abstain),
            );
            format!("{:>16}", f(m))
        })
        .collect();
    let _ = writeln!(o, "{:<12}{cells}\n", "false-abst");
    let _ = writeln!(
        o,
        "privacy leak_count (public reader, must be 0): {}  (unfiltered context: {}; reimpl_fusion not counted)",
        res.leak_count, res.leak_count_unfiltered
    );
    let _ = writeln!(
        o,
        "\ncandidate queries below gate ({}):",
        res.failures.len()
    );
    for x in &res.failures {
        let tag = if x.expected_fail {
            format!("  [expected_fail: {EXPECTED_FAIL_REASON}]")
        } else {
            String::new()
        };
        let _ = writeln!(
            o,
            "  {:<16}{:<16}{:.3} < gate {:.3}{tag}",
            x.id, x.metric, x.value, x.gate
        );
    }
    if !res.expected_fail_passing_today.is_empty() {
        let _ = writeln!(
            o,
            "as_of queries that already pass (floored, not flagged): {}",
            res.expected_fail_passing_today.join(", ")
        );
    }
    o
}
