//! `skmemory.validity.normalize_as_of`: parse an `as_of` instant in Rust so a
//! value Python rejects (`now`, `yesterday`, `infinity`, `2026-13-01`) is a
//! typed error here too, instead of reaching PostgreSQL's more lenient
//! `timestamptz` cast (SKMEM-R-04).
//!
//! Accepted, as Python 3.11+ `datetime.fromisoformat` does for the common
//! shapes: `YYYY-MM-DD` or `YYYYMMDD`, optionally followed by `T`, `t` or a
//! space and `HH[:MM[:SS[.f+]]]` (or compact `HHMM[SS]`), optionally followed by
//! `Z`/`z` or `+HH[:MM[:SS[.f+]]]`. A bare date is its 00:00 and a naive time is
//! UTC. Not accepted (Python would): ISO week dates and arbitrary separator
//! characters, a space before the offset, or a bare `Z` after a date; those
//! fail closed as a typed error (stricter than Python, never looser).

use crate::util::iso_from_unix;
use std::fmt;

/// The text was not an ISO-8601 timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadAsOf(pub String);

impl fmt::Display for BadAsOf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "as_of is not an ISO-8601 timestamp: {:?}", self.0)
    }
}

impl std::error::Error for BadAsOf {}

fn digits(s: &[u8], n: usize) -> Option<(u32, &[u8])> {
    if s.len() < n || !s[..n].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let v = s[..n].iter().fold(0u32, |a, d| a * 10 + (d - b'0') as u32);
    Some((v, &s[n..]))
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
    }
}

/// Days since 1970-01-01 (Hinnant's days-from-civil).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(micros, rest)` of a `[.f+]` fraction, cut to 6 digits like Python.
fn fraction(s: &[u8]) -> Option<(u32, &[u8])> {
    let Some(rest) = s.strip_prefix(b".").or_else(|| s.strip_prefix(b",")) else {
        return Some((0, s));
    };
    let n = rest.iter().take_while(|c| c.is_ascii_digit()).count();
    if n == 0 {
        return None;
    }
    let mut micros = 0u32;
    for i in 0..6 {
        micros = micros * 10
            + rest
                .get(i)
                .filter(|_| i < n)
                .map_or(0, |d| (d - b'0') as u32);
    }
    Some((micros, &rest[n..]))
}

/// Parse and normalise to Python's canonical UTC `isoformat()`. `Ok(None)` for
/// an empty or whitespace-only value (no filter); `Err` for anything else
/// that is not a timestamp.
pub fn normalize_as_of(value: &str) -> Result<Option<String>, BadAsOf> {
    let text = crate::model::py_strip(value);
    if text.is_empty() {
        return Ok(None);
    }
    parse(text)
        .map(Some)
        .ok_or_else(|| BadAsOf(value.to_string()))
}

fn parse(text: &str) -> Option<String> {
    let b = text.as_bytes();
    if !text.is_ascii() {
        return None;
    }
    let (y, rest) = digits(b, 4)?;
    let dash = rest.first() == Some(&b'-');
    let rest = if dash { &rest[1..] } else { rest };
    let (mo, rest) = digits(rest, 2)?;
    let rest = if dash { rest.strip_prefix(b"-")? } else { rest };
    let (d, rest) = digits(rest, 2)?;
    let y = y as i64;
    if y < 1 || !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) {
        return None;
    }
    let (mut h, mut mi, mut s, mut us) = (0u32, 0u32, 0u32, 0u32);
    let mut offset: i64 = 0;
    if !rest.is_empty() {
        if !matches!(rest[0], b'T' | b't' | b' ') {
            return None;
        }
        let t = &rest[1..];
        // Split the clock from an offset suffix at the first Z, z, + or -.
        let cut = t
            .iter()
            .position(|c| matches!(c, b'Z' | b'z' | b'+' | b'-'))
            .unwrap_or(t.len());
        let (clk, tz) = t.split_at(cut);
        let colon = clk.get(2) == Some(&b':');
        let (hh, c) = digits(clk, 2)?;
        h = hh;
        let mut c = c;
        let part = |c: &mut &[u8]| -> Option<Option<u32>> {
            if c.is_empty() || c[0] == b'.' || c[0] == b',' {
                return Some(None);
            }
            let cur: &[u8] = c;
            let r = if colon { cur.strip_prefix(b":")? } else { cur };
            let (v, r) = digits(r, 2)?;
            *c = r;
            Some(Some(v))
        };
        if let Some(v) = part(&mut c)? {
            mi = v;
            if let Some(v) = part(&mut c)? {
                s = v;
                let (f, r) = fraction(c)?;
                us = f;
                c = r;
            }
        }
        if !c.is_empty() {
            return None;
        }
        if h > 23 || mi > 59 || s > 59 {
            return None;
        }
        offset = parse_offset(tz)?;
    }
    let days = days_from_civil(y, mo, d);
    let secs = days * 86_400 + (h * 3600 + mi * 60 + s) as i64 - offset;
    // Python's aware datetimes stop at years 1..=9999.
    if !(-62_135_596_800..=253_402_300_799).contains(&secs) {
        return None;
    }
    Some(iso_from_unix(secs, us))
}

/// `""`, `Z`, `z`, or `+/-HH[:MM[:SS[.f]]]` (also compact) as seconds east of UTC.
fn parse_offset(tz: &[u8]) -> Option<i64> {
    if tz.is_empty() || tz == b"Z" || tz == b"z" {
        return Some(0);
    }
    let sign = match tz[0] {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let t = &tz[1..];
    let colon = t.get(2) == Some(&b':');
    let (hh, mut c) = digits(t, 2)?;
    let (mut mm, mut ss) = (0, 0);
    if !c.is_empty() {
        let r = if colon { c.strip_prefix(b":")? } else { c };
        let (v, r) = digits(r, 2)?;
        mm = v;
        c = r;
        if !c.is_empty() {
            let r = if colon { c.strip_prefix(b":")? } else { c };
            let (v, r) = digits(r, 2)?;
            ss = v;
            let (_, r) = fraction(r)?;
            c = r;
        }
    }
    // Python only requires the whole offset to be under 24 hours, so +05:60
    // is accepted (as 06:00).
    let total = (hh * 3600 + mm * 60 + ss) as i64;
    if !c.is_empty() || total >= 86_400 {
        return None;
    }
    Some(sign * total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let n = |s: &str| normalize_as_of(s).unwrap().unwrap();
        assert_eq!(n("2026-04-10"), "2026-04-10T00:00:00+00:00");
        assert_eq!(n("20260410"), "2026-04-10T00:00:00+00:00");
        assert_eq!(n("2026-04-10T05:00:00+05:00"), "2026-04-10T00:00:00+00:00");
        assert_eq!(
            n("2026-04-10 05:00:00.5Z"),
            "2026-04-10T05:00:00.500000+00:00"
        );
        assert_eq!(n("2026-04-10T05"), "2026-04-10T05:00:00+00:00");
        assert_eq!(normalize_as_of("  ").unwrap(), None);
        for bad in [
            "now",
            "yesterday",
            "infinity",
            "2026-13-01",
            "2026-02-30",
            "2026-04-10T24:00",
            "x",
        ] {
            assert!(normalize_as_of(bad).is_err(), "{bad}");
        }
    }
}
