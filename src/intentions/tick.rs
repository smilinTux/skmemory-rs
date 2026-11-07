//! Time tick for intentions: due time intentions fire into `sk-alert`
//! (SKMEM-R-07), the Rust twin of `skmemory/intention_tick.py`.
//!
//! The rules are the Python's and are load-bearing:
//!
//! * **Deliver first, then record the fire.** The alert goes through
//!   `sk-alert` and only when that exits 0 is the state change applied (via
//!   [`IntentionStore::apply`], the only state-change path). A crash between
//!   the two leaves the intention due, so the next tick delivers it again: an
//!   alert is never lost, a crash causes at most a duplicate.
//! * **Delivery failure** (non-zero exit, missing command, timeout) changes
//!   nothing and stops the tick: the intention stays eligible and is retried.
//! * **One tick host.** Only the host whose short, case-folded hostname equals
//!   `SKMEMORY_TICK_HOST` (or the `tick_host` argument) runs the tick; with
//!   none configured the tick refuses unless `any_host` is set.
//! * **Alert text** is only the intention title and the origin memory id plus
//!   a resolve hint. No memory content, scope or terms: channels may be shared.
//! * A host-local, non-blocking flock (`_tick.lock` in the store's lock
//!   directory, the same file the Python tick takes) serialises two ticks on
//!   one host, whichever engine runs them.
//! * The delivery command is `argv` from `SKMEMORY_ALERT_CMD` (default
//!   `sk-alert`) split shell-style but executed with no shell, 30 s timeout.

use super::{warn, Intention, IntentionStore, Kind, State};
use crate::model::py_strip;
use crate::pytime::Dt;
use crate::sys;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const ALERT_GROUP: &str = "intentions";
pub const ALERT_CMD_ENV: &str = "SKMEMORY_ALERT_CMD";
pub const TICK_HOST_ENV: &str = "SKMEMORY_TICK_HOST";
const MAX_TITLE: usize = 200;
pub const DEFAULT_MAX_PER_TICK: usize = 5;

/// What one tick did. `skipped` is set when the tick did not run at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickResult {
    pub skipped: Option<String>,
    pub fired: Vec<String>,
    pub failed: Vec<String>,
    /// Due but not delivered this tick (cap hit or stopped on failure).
    pub remaining: usize,
}

/// Python `str.isprintable` for one character.
fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    let cp = c as u32;
    crate::printable_table::NONPRINTABLE
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

/// One line, no control characters, bounded length (`_clean`).
pub fn clean(text: &str, limit: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if is_printable(c) { c } else { ' ' })
        .collect();
    let flat = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= limit {
        flat
    } else {
        let keep: String = flat.chars().take(limit.saturating_sub(3)).collect();
        format!("{keep}...")
    }
}

/// Title + origin memory id + one-line resolve hint. Nothing else leaves the host.
pub fn alert_text(i: &Intention) -> String {
    let origin = i.origin_memory_id.as_deref().unwrap_or("none");
    format!(
        "Intention due: {} (origin: {}). Resolve: skmemory intentions done {}",
        clean(&i.title, MAX_TITLE),
        clean(origin, 128),
        i.id
    )
}

/// POSIX `shlex.split` (no comments). `None` on an unterminated quote or
/// trailing escape, where Python raises `ValueError`.
pub fn shlex_split(s: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut have = false;
    let mut it = s.chars();
    while let Some(c) = it.next() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                if have {
                    out.push(std::mem::take(&mut cur));
                    have = false;
                }
            }
            '\\' => {
                cur.push(it.next()?);
                have = true;
            }
            '\'' => {
                have = true;
                loop {
                    match it.next()? {
                        '\'' => break,
                        ch => cur.push(ch),
                    }
                }
            }
            '"' => {
                have = true;
                loop {
                    match it.next()? {
                        '"' => break,
                        '\\' => {
                            let n = it.next()?;
                            if !matches!(n, '"' | '\\') {
                                cur.push('\\');
                            }
                            cur.push(n);
                        }
                        ch => cur.push(ch),
                    }
                }
            }
            ch => {
                cur.push(ch);
                have = true;
            }
        }
    }
    if have {
        out.push(cur);
    }
    Some(out)
}

/// Run `sk-alert -g intentions --title ... TEXT`; true only on exit 0. The
/// command is an argv list with no shell; a missing command, a non-zero exit
/// or a timeout is `false`.
pub fn sk_alert_deliver(text: &str, timeout: Duration) -> bool {
    let cmd = std::env::var(ALERT_CMD_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "sk-alert".into());
    let Some(mut argv) = shlex_split(&cmd).filter(|a| !a.is_empty()) else {
        warn("sk-alert delivery failed to run: bad SKMEMORY_ALERT_CMD");
        return false;
    };
    let prog = argv.remove(0);
    let mut child = match Command::new(&prog)
        .args(&argv)
        .args(["-g", ALERT_GROUP, "--title", "skmemory intention due", text])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            warn(&format!("sk-alert delivery failed to run: {e}"));
            return false;
        }
    };
    // Drain stderr on a thread (bounded copy kept) so a chatty child never blocks on a full pipe.
    let captured = Arc::new(Mutex::new(Vec::<u8>::new()));
    let reader = child.stderr.take().map(|mut p| {
        let cap = Arc::clone(&captured);
        std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            while let Ok(n) = p.read(&mut buf) {
                if n == 0 {
                    break;
                }
                if let Ok(mut c) = cap.lock() {
                    let room = 4096usize.saturating_sub(c.len());
                    c.extend_from_slice(&buf[..n.min(room)]);
                }
            }
        })
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break None,
        }
    };
    // An orphaned grandchild can hold the pipe open: wait briefly, never forever.
    if let Some(h) = reader {
        let end = Instant::now() + Duration::from_millis(200);
        while !h.is_finished() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    let stderr = captured
        .lock()
        .map(|c| String::from_utf8_lossy(&c).into_owned())
        .unwrap_or_default();
    match status {
        None => {
            warn("sk-alert delivery failed to run: timed out");
            false
        }
        Some(st) if st.success() => true,
        Some(st) => {
            warn(&format!(
                "sk-alert exited {}: {}",
                st.code().unwrap_or(-1),
                stderr.trim().chars().take(200).collect::<String>()
            ));
            false
        }
    }
}

/// Whether a time intention is due at `now`.
pub fn is_due(i: &Intention, now: Dt) -> bool {
    i.kind == Kind::Time
        && matches!(i.state, State::Pending | State::Armed)
        && i.due_at.is_some_and(|d| d <= now)
        && i.snooze_until.is_none_or(|s| s <= now)
}

/// Due time intentions, judged from the flat files (truth), oldest due first.
pub fn due_intentions(store: &IntentionStore, now: Dt) -> super::Result<Vec<Intention>> {
    let mut out = Vec::new();
    for cand in store.list(None)? {
        if let Some(cur) = store.get(&cand.id)? {
            if is_due(&cur, now) {
                out.push(cur);
            }
        }
    }
    out.sort_by(|a, b| (a.due_at, &a.id).cmp(&(b.due_at, &b.id)));
    Ok(out)
}

/// `str.casefold` for the short host name: lowercasing plus the full case
/// foldings that differ from it for letters a hostname can hold.
fn casefold(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            'ß' | 'ẞ' => o.push_str("ss"),
            'ς' => o.push('σ'),
            'ſ' => o.push('s'),
            c => o.extend(c.to_lowercase()),
        }
    }
    o
}

fn short_host(h: &str) -> String {
    casefold(py_strip(h).split('.').next().unwrap_or(""))
}

/// Inputs of [`run_tick`].
pub struct TickOptions<'a> {
    pub now: Option<Dt>,
    /// Delivery; default [`sk_alert_deliver`] with a 30 s timeout.
    pub deliver: Option<&'a mut dyn FnMut(&str) -> bool>,
    pub tick_host: Option<&'a str>,
    pub any_host: bool,
    pub hostname: Option<&'a str>,
    pub max_per_tick: usize,
}

impl Default for TickOptions<'_> {
    fn default() -> Self {
        TickOptions {
            now: None,
            deliver: None,
            tick_host: None,
            any_host: false,
            hostname: None,
            max_per_tick: DEFAULT_MAX_PER_TICK,
        }
    }
}

/// Fire every due time intention once. See the module docs for the rules.
pub fn run_tick(store: &IntentionStore, opts: TickOptions<'_>) -> super::Result<TickResult> {
    let now = opts.now.unwrap_or_else(Dt::now);
    let mut default_deliver = |t: &str| sk_alert_deliver(t, Duration::from_secs(30));
    let deliver: &mut dyn FnMut(&str) -> bool = match opts.deliver {
        Some(d) => d,
        None => &mut default_deliver,
    };
    let host = short_host(&opts.hostname.map_or_else(sys::hostname, str::to_string));
    let want = short_host(
        &opts
            .tick_host
            .map(str::to_string)
            .or_else(|| std::env::var(TICK_HOST_ENV).ok())
            .unwrap_or_default(),
    );
    if !want.is_empty() {
        if want != host {
            return Ok(TickResult {
                skipped: Some(format!(
                    "not the tick host (this host {}, tick host {})",
                    super::rp(&host),
                    super::rp(&want)
                )),
                ..Default::default()
            });
        }
    } else if !opts.any_host {
        return Ok(TickResult {
            skipped: Some(format!(
                "no tick host configured: set {TICK_HOST_ENV} or pass --tick-host/--any-host"
            )),
            ..Default::default()
        });
    }

    let mut res = TickResult::default();
    let Ok(_lock) = store.lock_named("_tick.lock", Some(0.0)) else {
        res.skipped = Some("another tick is running on this host".into());
        return Ok(res);
    };
    store.reindex()?; // merge synced replicas before judging what is due
    let due = due_intentions(store, now)?; // oldest due first
    let cap = opts.max_per_tick.max(1);
    let (batch, rest) = due.split_at(cap.min(due.len()));
    for i in batch {
        if !deliver(&alert_text(i)) {
            // Stop on the first failure: sk-alert is down, retrying every item
            // would take N x timeout and overlap the next run.
            warn(&format!(
                "intention {}: alert not delivered, stopping tick",
                i.id
            ));
            res.failed.push(i.id.clone());
            res.remaining = due.len() - res.fired.len();
            return Ok(res);
        }
        // only after delivery succeeded
        match store.apply(&i.id, "fire", Some(now), None)? {
            None => {
                warn(&format!(
                    "intention {} delivered but could not be recorded",
                    i.id
                ));
                res.failed.push(i.id.clone());
            }
            Some(_) => res.fired.push(i.id.clone()),
        }
    }
    res.remaining = rest.len();
    if !rest.is_empty() && !deliver(&format!("{} more due", rest.len())) {
        warn(&format!(
            "summary alert for {} more due not delivered",
            rest.len()
        ));
    }
    Ok(res)
}
