//! A timezone-aware instant with Python `datetime` semantics, std only.
//!
//! The intention store and the outcome ledger carry Python `datetime` values
//! through JSON, so the offset a writer chose has to survive a rewrite: a
//! [`Dt`] is a UTC microsecond count plus the offset it is displayed in.
//! Equality and ordering look at the instant only (as aware datetimes do).
//!
//! Two text forms are needed and both are here:
//!
//! * [`Dt::isoformat`] is `datetime.isoformat()` (`+00:00` for UTC), used in
//!   SQLite columns and ledger lines;
//! * [`Dt::pydantic`] is what pydantic's JSON writer emits (`Z` for UTC),
//!   used in the flat intention files.
//!
//! A naive input is taken as UTC, as `_aware` does in the Python.

use std::cmp::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

const US: i64 = 1_000_000;

/// An aware instant: UTC microseconds since the epoch plus a display offset.
#[derive(Clone, Copy, Debug)]
pub struct Dt {
    /// Microseconds since 1970-01-01T00:00:00Z.
    pub micros: i64,
    /// Display offset from UTC in seconds (0 for UTC).
    pub offset_s: i32,
}

impl PartialEq for Dt {
    fn eq(&self, o: &Self) -> bool {
        self.micros == o.micros
    }
}
impl Eq for Dt {}
impl PartialOrd for Dt {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Dt {
    fn cmp(&self, o: &Self) -> Ordering {
        self.micros.cmp(&o.micros)
    }
}

/// The Unix epoch, the merge key's "never fired" value.
pub const EPOCH: Dt = Dt {
    micros: 0,
    offset_s: 0,
};

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

impl Dt {
    /// A UTC instant from microseconds since the epoch.
    pub fn from_micros(micros: i64) -> Dt {
        Dt {
            micros,
            offset_s: 0,
        }
    }

    /// The current time, UTC.
    pub fn now() -> Dt {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Dt::from_micros(d.as_secs() as i64 * US + i64::from(d.subsec_micros()))
    }

    /// The same instant displayed in UTC (`astimezone(timezone.utc)`).
    pub fn to_utc(self) -> Dt {
        Dt::from_micros(self.micros)
    }

    /// This instant moved by `delta` microseconds, keeping the display offset
    /// (`datetime + timedelta`).
    pub fn add_micros(self, delta: i64) -> Dt {
        Dt {
            micros: self.micros + delta,
            offset_s: self.offset_s,
        }
    }

    /// Calendar fields in the display offset: year, month, day, hour, minute, second, microsecond.
    pub fn fields(self) -> (i64, i64, i64, i64, i64, i64, i64) {
        let local = self.micros + i64::from(self.offset_s) * US;
        let secs = local.div_euclid(US);
        let us = local.rem_euclid(US);
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        (y, m, d, rem / 3600, rem % 3600 / 60, rem % 60, us)
    }

    /// Parse an ISO 8601 timestamp the way `datetime.fromisoformat` (after
    /// the `Z` to `+00:00` swap) and pydantic accept it. A naive value is UTC.
    pub fn parse(s: &str) -> Option<Dt> {
        let b = s.as_bytes();
        if !s.is_ascii() || b.len() < 10 {
            return None;
        }
        let num = |a: usize, z: usize| -> Option<i64> {
            let t = s.get(a..z)?;
            if t.bytes().all(|c| c.is_ascii_digit()) && !t.is_empty() {
                t.parse().ok()
            } else {
                None
            }
        };
        let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
        if b[4] != b'-' || b[7] != b'-' || !(1..=12).contains(&mo) || y < 1 {
            return None;
        }
        if d < 1 || d > days_in_month(y, mo) {
            return None;
        }
        let (mut hh, mut mi, mut ss, mut us) = (0i64, 0i64, 0i64, 0i64);
        let mut off = 0i32;
        let mut i = 10;
        if i < b.len() {
            if !matches!(b[i], b'T' | b't' | b' ') {
                return None;
            }
            i += 1;
            hh = num(i, i + 2)?;
            i += 2;
            if i < b.len() && b[i] == b':' {
                mi = num(i + 1, i + 3)?;
                i += 3;
                if i < b.len() && b[i] == b':' {
                    ss = num(i + 1, i + 3)?;
                    i += 3;
                    if i < b.len() && (b[i] == b'.' || b[i] == b',') {
                        let st = i + 1;
                        let mut e = st;
                        while e < b.len() && b[e].is_ascii_digit() {
                            e += 1;
                        }
                        if e == st {
                            return None;
                        }
                        let mut frac = s[st..e].to_string();
                        frac.truncate(6);
                        while frac.len() < 6 {
                            frac.push('0');
                        }
                        us = frac.parse().ok()?;
                        i = e;
                    }
                }
            }
            if hh > 23 || mi > 59 || ss > 59 {
                return None;
            }
            if i < b.len() {
                match b[i] {
                    b'Z' | b'z' if i + 1 == b.len() => {}
                    b'+' | b'-' => {
                        let sign = if b[i] == b'-' { -1 } else { 1 };
                        let rest = &s[i + 1..];
                        let (oh, om, os) = match rest.len() {
                            2 => (num(i + 1, i + 3)?, 0, 0),
                            4 => (num(i + 1, i + 3)?, num(i + 3, i + 5)?, 0),
                            5 if b[i + 3] == b':' => (num(i + 1, i + 3)?, num(i + 4, i + 6)?, 0),
                            8 if b[i + 3] == b':' && b[i + 6] == b':' => {
                                (num(i + 1, i + 3)?, num(i + 4, i + 6)?, num(i + 7, i + 9)?)
                            }
                            _ => return None,
                        };
                        if oh > 23 || om > 59 || os > 59 {
                            return None;
                        }
                        off = sign * (oh * 3600 + om * 60 + os) as i32;
                    }
                    _ => return None,
                }
            }
        }
        let days = days_from_civil(y, mo, d);
        let local = (days * 86_400 + hh * 3600 + mi * 60 + ss) * US + us;
        Some(Dt {
            micros: local - i64::from(off) * US,
            offset_s: off,
        })
    }

    fn base(self) -> String {
        let (y, m, d, h, mi, s, us) = self.fields();
        let mut out = format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}");
        if us != 0 {
            out.push_str(&format!(".{us:06}"));
        }
        out
    }

    fn offset_text(self, utc: &str) -> String {
        if self.offset_s == 0 {
            return utc.to_string();
        }
        let a = self.offset_s.abs();
        let sign = if self.offset_s < 0 { '-' } else { '+' };
        if a % 60 != 0 {
            format!("{sign}{:02}:{:02}:{:02}", a / 3600, a % 3600 / 60, a % 60)
        } else {
            format!("{sign}{:02}:{:02}", a / 3600, a % 3600 / 60)
        }
    }

    /// `datetime.isoformat()`: `2026-11-01T12:00:00+00:00`, fraction only when non-zero.
    pub fn isoformat(self) -> String {
        format!("{}{}", self.base(), self.offset_text("+00:00"))
    }

    /// Pydantic's JSON form: like [`Dt::isoformat`] but UTC is `Z`.
    pub fn pydantic(self) -> String {
        format!("{}{}", self.base(), self.offset_text("Z"))
    }

    /// `strftime("%Y-%m")` of the UTC instant.
    pub fn year_month_utc(self) -> String {
        let (y, m, ..) = self.to_utc().fields();
        format!("{y:04}-{m:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        for s in [
            "2026-11-01T12:00:00+00:00",
            "2026-11-01T12:00:00.5+02:00",
            "2026-11-01 12:00:00Z",
            "2026-11-01",
        ] {
            assert!(Dt::parse(s).is_some(), "{s}");
        }
        let d = Dt::parse("2026-11-06T12:00:00.5Z").unwrap();
        assert_eq!(d.pydantic(), "2026-11-06T12:00:00.500000Z");
        assert_eq!(d.isoformat(), "2026-11-06T12:00:00.500000+00:00");
        let d = Dt::parse("2026-11-06T12:00:00+02:00").unwrap();
        assert_eq!(d.pydantic(), "2026-11-06T12:00:00+02:00");
        assert_eq!(d.to_utc().isoformat(), "2026-11-06T10:00:00+00:00");
        assert!(Dt::parse("2026-13-01").is_none());
        assert!(Dt::parse("2026-02-30").is_none());
        assert!(Dt::parse("nope").is_none());
        assert_eq!(
            Dt::parse("2026-11-01T12:00:00+00:00").unwrap().micros,
            1_793_534_400_000_000
        );
    }
}
