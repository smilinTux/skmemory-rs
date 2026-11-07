//! Embedding client, the Rust twin of `PGVectorBackend._embed` and
//! `skmemory.embed_text` (SKMEM-R-04).
//!
//! Configuration comes from the same environment as Python:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `SKMEMORY_EMBED_URL` | primary endpoint | `http://localhost:11434/api/embed` |
//! | `SKMEMORY_EMBED_URLS` | comma separated failover list | the primary alone |
//! | `SKMEMORY_EMBED_MODEL` | pinned model | `mxbai-embed-large` |
//! | `SKMEMORY_EMBED_TIMEOUT` | per read seconds | 15 |
//! | `SKMEMORY_EMBED_CONNECT_TIMEOUT` | connect seconds | `min(5, timeout)` |
//! | `SKMEMORY_EMBED_VERIFY` | `0/false/no/off` disables the identity check | on |
//!
//! The client is std-only: a small HTTP/1.1 POST over `TcpStream`. Only
//! `http://` endpoints are supported (the embed server is a LAN Ollama);
//! an `https://` endpoint counts as a failed endpoint and failover moves on.

use crate::model::py_strip;
use serde_json::{json, Value};
use std::fmt;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Vector dimension of the fleet model (mxbai-embed-large).
pub const VECTOR_DIM: usize = 1024;
/// Characters of embed text sent to the model.
pub const EMBED_TEXT_MAX_CHARS: usize = 1400;
pub const DEFAULT_EMBED_URL: &str = "http://localhost:11434/api/embed";
pub const DEFAULT_EMBED_MODEL: &str = "mxbai-embed-large";
const DEFAULT_TIMEOUT: f64 = 15.0;
const MAX_BODY: usize = 64 << 20;

/// `skmemory.embed_text.embed_text`: `title\ncontent\nsummary`, stripped (Python
/// `str.strip`), cut at 1400 characters (code points).
pub fn embed_text(title: &str, content: &str, summary: &str) -> String {
    let joined = format!("{title}\n{content}\n{summary}");
    py_strip(&joined)
        .chars()
        .take(EMBED_TEXT_MAX_CHARS)
        .collect()
}

/// Why an embedding could not be produced. `Display` never carries a DSN.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedError {
    /// The endpoint returned a vector of the wrong length (a different model
    /// is being served). Never retried: every endpoint would do the same.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        endpoint: String,
    },
    /// The endpoint reports serving a different model than the pinned one.
    ModelMismatch {
        expected: String,
        served: String,
        endpoint: String,
    },
    /// Write path only: every endpoint failed or returned an empty or
    /// all-zero vector.
    Unavailable { endpoints: Vec<String> },
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::DimensionMismatch {
                expected,
                actual,
                endpoint,
            } => write!(
                f,
                "embedding dimension mismatch: expected {expected} but {endpoint} returned {actual}; set SKMEMORY_EMBED_VERIFY=0 to override"
            ),
            EmbedError::ModelMismatch {
                expected,
                served,
                endpoint,
            } => write!(
                f,
                "embedding model mismatch: expected {expected:?} but {endpoint} reported serving {served:?}"
            ),
            EmbedError::Unavailable { endpoints } => write!(
                f,
                "embedding unavailable: every configured endpoint ({}) was down, timed out, or returned an empty vector",
                endpoints.join(", ")
            ),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Anything that turns text into a query vector (the Python `embed_fn`).
pub trait Embedder {
    /// An empty vector means "no embedding" when `required` is false.
    fn embed(&self, text: &str, required: bool) -> Result<Vec<f32>, EmbedError>;
}

/// Client settings, resolved like the Python module globals.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedConfig {
    pub urls: Vec<String>,
    pub model: String,
    pub dim: usize,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub verify: bool,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        EmbedConfig {
            urls: vec![DEFAULT_EMBED_URL.to_string()],
            model: DEFAULT_EMBED_MODEL.to_string(),
            dim: VECTOR_DIM,
            timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT),
            connect_timeout: Duration::from_secs_f64(5.0),
            verify: true,
        }
    }
}

/// `_parse_embed_urls`: comma separated, blanks dropped, first-wins dedup, the
/// primary when nothing usable is given.
pub fn parse_embed_urls(raw: Option<&str>, primary: &str) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    for u in raw.unwrap_or("").split(',') {
        let u = py_strip(u);
        if !u.is_empty() && !urls.iter().any(|x| x == u) {
            urls.push(u.to_string());
        }
    }
    if urls.is_empty() && !primary.is_empty() {
        urls.push(primary.to_string());
    }
    urls
}

fn positive_float(raw: Option<String>, default: f64) -> f64 {
    match raw {
        Some(r) if !py_strip(&r).is_empty() => match py_strip(&r).parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => v,
            _ => default,
        },
        _ => default,
    }
}

impl EmbedConfig {
    /// Resolve from the process environment.
    pub fn from_env() -> EmbedConfig {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Resolve from any key lookup (tests pass a map instead of mutating the
    /// process environment).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> EmbedConfig {
        let primary = get("SKMEMORY_EMBED_URL").unwrap_or_else(|| DEFAULT_EMBED_URL.to_string());
        let urls = parse_embed_urls(get("SKMEMORY_EMBED_URLS").as_deref(), &primary);
        let timeout = positive_float(get("SKMEMORY_EMBED_TIMEOUT"), DEFAULT_TIMEOUT);
        let connect = positive_float(get("SKMEMORY_EMBED_CONNECT_TIMEOUT"), timeout.min(5.0));
        let verify = !matches!(
            get("SKMEMORY_EMBED_VERIFY")
                .unwrap_or_else(|| "1".to_string())
                .to_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        );
        EmbedConfig {
            urls,
            model: get("SKMEMORY_EMBED_MODEL").unwrap_or_else(|| DEFAULT_EMBED_MODEL.to_string()),
            dim: VECTOR_DIM,
            timeout: Duration::from_secs_f64(timeout),
            connect_timeout: Duration::from_secs_f64(connect),
            verify,
        }
    }
}

fn normalize_model_name(name: &str) -> String {
    let base = py_strip(name).to_lowercase();
    let base = base.split(':').next().unwrap_or("").to_string();
    base.rsplit('/').next().unwrap_or("").to_string()
}

/// `_model_names_match`: the shorter normalised name must sit inside the longer.
pub fn model_names_match(expected: &str, actual: &str) -> bool {
    let exp = normalize_model_name(expected);
    let act = normalize_model_name(actual);
    if exp.is_empty() || act.is_empty() {
        return true;
    }
    let (short, long) = if exp.chars().count() <= act.chars().count() {
        (&exp, &act)
    } else {
        (&act, &exp)
    };
    long.contains(short.as_str())
}

/// The HTTP embedding client with endpoint failover.
#[derive(Clone, Debug)]
pub struct EmbedClient {
    pub config: EmbedConfig,
}

/// Outcome of one endpoint attempt.
enum One {
    Vec(Vec<f32>),
    /// Transient failure or unusable payload: try the next endpoint.
    Skip,
}

impl EmbedClient {
    pub fn new(config: EmbedConfig) -> EmbedClient {
        EmbedClient { config }
    }

    pub fn from_env() -> EmbedClient {
        EmbedClient::new(EmbedConfig::from_env())
    }

    fn verify_vec(
        &self,
        vec: Vec<f32>,
        served: Option<&Value>,
        url: &str,
    ) -> Result<Option<Vec<f32>>, EmbedError> {
        if !self.config.verify || vec.is_empty() {
            return Ok(Some(vec));
        }
        if vec.len() != self.config.dim {
            return Err(EmbedError::DimensionMismatch {
                expected: self.config.dim,
                actual: vec.len(),
                endpoint: url.to_string(),
            });
        }
        match served {
            Some(Value::String(s)) if !s.is_empty() => {
                if !model_names_match(&self.config.model, s) {
                    return Err(EmbedError::ModelMismatch {
                        expected: self.config.model.clone(),
                        served: s.clone(),
                        endpoint: url.to_string(),
                    });
                }
            }
            // Python would raise AttributeError on a truthy non-string and
            // treat the endpoint as failed.
            Some(Value::Null) | None | Some(Value::String(_)) => {}
            Some(Value::Bool(false)) => {}
            Some(Value::Number(n)) if n.as_f64() == Some(0.0) => {}
            Some(Value::Array(a)) if a.is_empty() => {}
            Some(Value::Object(o)) if o.is_empty() => {}
            Some(_) => return Ok(None),
        }
        Ok(Some(vec))
    }

    fn embed_one(&self, url: &str, text: &str) -> Result<One, EmbedError> {
        let mut text: String = text.to_string();
        while !text.is_empty() {
            let body =
                json!({"model": self.config.model, "input": text, "truncate": true}).to_string();
            match post_json(url, &body, self.config.connect_timeout, self.config.timeout) {
                Ok((status, resp)) => {
                    if !(200..300).contains(&status) {
                        let n = text.chars().count();
                        if status == 400 && n > 200 {
                            text = text.chars().take(n / 2).collect();
                            continue;
                        }
                        return Ok(One::Skip);
                    }
                    let Ok(data) = serde_json::from_slice::<Value>(&resp) else {
                        return Ok(One::Skip);
                    };
                    let Some(obj) = data.as_object() else {
                        return Ok(One::Skip);
                    };
                    let served = obj.get("model");
                    let raw = if let Some(e) = obj.get("embeddings") {
                        e.get(0)
                    } else if let Some(d) = obj.get("data") {
                        d.get(0).and_then(|x| x.get("embedding"))
                    } else if let Some(e) = obj.get("embedding") {
                        Some(e)
                    } else {
                        return Ok(One::Vec(Vec::new()));
                    };
                    let Some(vec) = raw.and_then(to_f32_vec) else {
                        return Ok(One::Skip);
                    };
                    return Ok(match self.verify_vec(vec, served, url)? {
                        Some(v) => One::Vec(v),
                        None => One::Skip,
                    });
                }
                Err(_) => return Ok(One::Skip),
            }
        }
        Ok(One::Vec(Vec::new()))
    }
}

fn to_f32_vec(v: &Value) -> Option<Vec<f32>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32))
        .collect()
}

impl Embedder for EmbedClient {
    /// Walk the endpoints in order and return the first usable vector.
    /// Model or dimension mismatches always propagate. With `required` (the
    /// write path) an empty or all-zero result is `Unavailable`.
    fn embed(&self, text: &str, required: bool) -> Result<Vec<f32>, EmbedError> {
        let text: String = text.chars().take(EMBED_TEXT_MAX_CHARS).collect();
        let mut vec: Vec<f32> = Vec::new();
        for url in &self.config.urls {
            if let One::Vec(v) = self.embed_one(url, &text)? {
                if !v.is_empty() {
                    vec = v;
                    break;
                }
            }
        }
        if required && (vec.is_empty() || vec.iter().all(|x| *x == 0.0)) {
            return Err(EmbedError::Unavailable {
                endpoints: self.config.urls.clone(),
            });
        }
        Ok(vec)
    }
}

struct Target {
    host: String,
    port: u16,
    path: String,
    host_header: String,
}

fn parse_http_url(url: &str) -> Result<Target, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| "only http:// endpoints are supported".to_string())?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let (host, port) = if let Some(stripped) = authority.strip_prefix('[') {
        let end = stripped.find(']').ok_or("bad IPv6 host")?;
        let host = &stripped[..end];
        let port = stripped[end + 1..].strip_prefix(':');
        (host.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (authority.to_string(), None),
        }
    };
    if host.is_empty() {
        return Err("empty host".into());
    }
    let port: u16 = match port {
        Some(p) => p.parse().map_err(|_| "bad port".to_string())?,
        None => 80,
    };
    let host_header = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    Ok(Target {
        host,
        port,
        path: path.to_string(),
        host_header,
    })
}

/// POST `body` as JSON and return `(status, body bytes)`.
fn post_json(
    url: &str,
    body: &str,
    connect: Duration,
    read: Duration,
) -> Result<(u16, Vec<u8>), String> {
    let t = parse_http_url(url)?;
    let addrs = (t.host.as_str(), t.port)
        .to_socket_addrs()
        .map_err(|e| e.to_string())?;
    let mut stream = None;
    let mut last = String::from("no address");
    for a in addrs {
        match TcpStream::connect_timeout(&a, connect) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last = e.to_string(),
        }
    }
    let mut s = stream.ok_or(last)?;
    s.set_read_timeout(Some(read)).map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(read)).map_err(|e| e.to_string())?;
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        t.path,
        t.host_header,
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    let mut buf = [0u8; 16384];
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if raw.len() > MAX_BODY {
                    return Err("response too large".into());
                }
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    parse_response(&raw)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn parse_response(raw: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let split = find(raw, b"\r\n\r\n").ok_or("truncated response")?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let body = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or("empty response")?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("bad status line")?;
    let mut chunked = false;
    let mut length: Option<usize> = None;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            let v = v.trim();
            if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            } else if k == "content-length" {
                length = v.parse().ok();
            }
        }
    }
    if chunked {
        return Ok((status, dechunk(body)?));
    }
    if let Some(n) = length {
        if body.len() < n {
            return Err("truncated body".into());
        }
        return Ok((status, body[..n].to_vec()));
    }
    Ok((status, body.to_vec()))
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let eol = find(b, b"\r\n").ok_or("bad chunk")?;
        let size_txt = String::from_utf8_lossy(&b[..eol]).to_string();
        let size_txt = size_txt.split(';').next().unwrap_or("").trim().to_string();
        let size = usize::from_str_radix(&size_txt, 16).map_err(|_| "bad chunk size")?;
        b = &b[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size + 2 {
            return Err("truncated chunk".into());
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size + 2..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_text_matches_python_formula() {
        assert_eq!(embed_text("T", "C", ""), "T\nC");
        assert_eq!(embed_text("T", "C", "S"), "T\nC\nS");
        assert_eq!(embed_text("", "", ""), "");
        assert_eq!(embed_text(" ", "x", ""), "x");
        let long = "é".repeat(2000);
        assert_eq!(embed_text("", &long, "").chars().count(), 1400);
    }

    #[test]
    fn urls_dedup_and_default() {
        assert_eq!(parse_embed_urls(None, "http://a/x"), vec!["http://a/x"]);
        assert_eq!(
            parse_embed_urls(Some(" http://b ,,http://c,http://b"), "http://a"),
            vec!["http://b", "http://c"]
        );
        assert_eq!(parse_embed_urls(Some(" , "), "http://a"), vec!["http://a"]);
    }

    #[test]
    fn config_from_env_values() {
        let env = |k: &str| match k {
            "SKMEMORY_EMBED_URLS" => Some("http://p:1/e,http://s:2/e".to_string()),
            "SKMEMORY_EMBED_TIMEOUT" => Some("3".into()),
            "SKMEMORY_EMBED_VERIFY" => Some("OFF".into()),
            "SKMEMORY_EMBED_MODEL" => Some("m".into()),
            _ => None,
        };
        let c = EmbedConfig::from_lookup(env);
        assert_eq!(c.urls.len(), 2);
        assert_eq!(c.timeout, Duration::from_secs(3));
        assert_eq!(c.connect_timeout, Duration::from_secs(3));
        assert!(!c.verify);
        assert_eq!(c.model, "m");
        let d = EmbedConfig::from_lookup(|k| (k == "SKMEMORY_EMBED_TIMEOUT").then(|| "-1".into()));
        assert_eq!(d.timeout, Duration::from_secs(15));
        assert_eq!(d.connect_timeout, Duration::from_secs(5));
        assert!(d.verify);
    }

    #[test]
    fn model_name_matching() {
        assert!(model_names_match(
            "mxbai-embed-large",
            "mxbai-embed-large:latest"
        ));
        assert!(model_names_match(
            "mxbai-embed-large",
            "mixedbread-ai/mxbai-embed-large-v1"
        ));
        assert!(!model_names_match("mxbai-embed-large", "nomic-embed-text"));
        assert!(model_names_match("", "anything"));
    }

    #[test]
    fn http_parsing() {
        let t = parse_http_url("http://[::1]:11434/api/embed").unwrap();
        assert_eq!(
            (t.host.as_str(), t.port, t.path.as_str()),
            ("::1", 11434, "/api/embed")
        );
        assert_eq!(parse_http_url("http://h").unwrap().path, "/");
        assert!(parse_http_url("https://h/x").is_err());
        let r = parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n")
            .unwrap();
        assert_eq!(r, (200, b"abcde".to_vec()));
    }
}
