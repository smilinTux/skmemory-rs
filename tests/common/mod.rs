//! Shared test helpers: the deterministic embedder and a tiny fake embed
//! server. The embedder is mirrored, byte for byte, by
//! `tests/pg_parity/gen_pg_parity.py` so Python and Rust vectors are equal.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

pub const DIM: usize = 1024;

fn fnv1a64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for c in b {
        h = (h ^ *c as u64).wrapping_mul(0x100_0000_01B3);
    }
    h
}

fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sum of per-token pseudo-random integer vectors, scaled by 1/64 (exact in
/// f64 and f32, so there is no rounding to disagree about).
pub fn det_embed(text: &str) -> Vec<f32> {
    let mut acc = vec![0i64; DIM];
    for tok in text.to_lowercase().split_whitespace() {
        let h = fnv1a64(tok.as_bytes());
        for (i, a) in acc.iter_mut().enumerate() {
            *a += ((splitmix64(h.wrapping_add(i as u64)) >> 33) % 1001) as i64 - 500;
        }
    }
    acc.into_iter().map(|a| (a as f64 / 64.0) as f32).collect()
}

/// How the fake server answers one request.
pub enum Reply {
    /// 200 with the deterministic vector in Ollama shape.
    Ok,
    /// 200 in OpenAI shape, chunked.
    OpenAiChunked,
    /// 200 with a vector of this length.
    WrongDim(usize),
    /// 200 reporting this served model name.
    Model(&'static str),
    /// 200 with an all-zero vector.
    Zeros,
    /// 400 when the input is longer than 300 chars, else Ok.
    Overflow400,
    /// A bare status code and no vector.
    Status(u16),
    /// Accept and never answer.
    Hang,
}

/// Serve `reply` forever on 127.0.0.1; returns the endpoint URL and a shared
/// log of the `input` lengths received.
pub fn serve(reply: Reply) -> (String, std::sync::Arc<std::sync::Mutex<Vec<usize>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/api/embed", l.local_addr().unwrap());
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log2 = log.clone();
    thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let (head_end, len) = loop {
                let n = s.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break (0, 0);
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    break (p + 4, len);
                }
            };
            while buf.len() < head_end + len {
                let n = s.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body: serde_json::Value =
                serde_json::from_slice(&buf[head_end..]).unwrap_or_default();
            let input = body["input"].as_str().unwrap_or("").to_string();
            log2.lock().unwrap().push(input.chars().count());
            let ollama = |vec: &[f32], model: &str| {
                serde_json::json!({"model": model, "embeddings": [vec]}).to_string()
            };
            let (status, payload, chunked) = match &reply {
                Reply::Ok => (200, ollama(&det_embed(&input), "mxbai-embed-large"), false),
                Reply::OpenAiChunked => (
                    200,
                    serde_json::json!({"data": [{"embedding": det_embed(&input)}]}).to_string(),
                    true,
                ),
                Reply::WrongDim(n) => (200, ollama(&vec![0.5; *n], "mxbai-embed-large"), false),
                Reply::Model(m) => (200, ollama(&det_embed(&input), m), false),
                Reply::Zeros => (200, ollama(&vec![0.0; DIM], "mxbai-embed-large"), false),
                Reply::Overflow400 => {
                    if input.chars().count() > 300 {
                        (400, "{}".to_string(), false)
                    } else {
                        (200, ollama(&det_embed(&input), "mxbai-embed-large"), false)
                    }
                }
                Reply::Status(c) => (*c, "{}".to_string(), false),
                Reply::Hang => {
                    thread::sleep(std::time::Duration::from_secs(5));
                    continue;
                }
            };
            let resp = if chunked {
                let half = payload.len() / 2;
                format!(
                    "HTTP/1.1 {status} X\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{}\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                    half,
                    &payload[..half],
                    payload.len() - half,
                    &payload[half..]
                )
            } else {
                format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                )
            };
            let _ = s.write_all(resp.as_bytes());
        }
    });
    (url, log)
}
