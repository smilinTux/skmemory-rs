//! SKMEM-R-04: the embedding client against a local fake server (hermetic).

mod common;

use common::{det_embed, serve, Reply, DIM};
use skmem::embed::{embed_text, EmbedClient, EmbedConfig, EmbedError, Embedder};
use std::path::Path;
use std::time::Duration;

fn client(urls: Vec<String>) -> EmbedClient {
    EmbedClient::new(EmbedConfig {
        urls,
        timeout: Duration::from_secs(2),
        connect_timeout: Duration::from_secs(1),
        ..EmbedConfig::default()
    })
}

const DEAD: &str = "http://127.0.0.1:1/api/embed";

#[test]
fn corpus_embed_text_matches_python() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pg_parity/corpus.json");
    let rows: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    assert!(rows.len() > 50);
    for r in rows {
        let got = embed_text(
            r["title"].as_str().unwrap(),
            r["content"].as_str().unwrap(),
            r["summary"].as_str().unwrap(),
        );
        assert_eq!(got, r["embed_text"].as_str().unwrap(), "{}", r["id"]);
    }
}

#[test]
fn happy_path_both_response_shapes() {
    let (u1, _) = serve(Reply::Ok);
    let v = client(vec![u1]).embed("River stone", true).unwrap();
    assert_eq!(v.len(), DIM);
    assert_eq!(v, det_embed("river stone"));
    let (u2, _) = serve(Reply::OpenAiChunked);
    assert_eq!(client(vec![u2]).embed("River stone", true).unwrap(), v);
}

#[test]
fn fails_over_to_the_next_endpoint() {
    let (good, _) = serve(Reply::Ok);
    let (bad5xx, _) = serve(Reply::Status(503));
    let c = client(vec![DEAD.into(), bad5xx, "https://x/y".into(), good]);
    assert_eq!(c.embed("a b", true).unwrap(), det_embed("a b"));
}

#[test]
fn hung_endpoint_times_out_and_fails_over() {
    let (hang, _) = serve(Reply::Hang);
    let (good, _) = serve(Reply::Ok);
    let c = EmbedClient::new(EmbedConfig {
        urls: vec![hang, good],
        timeout: Duration::from_millis(300),
        connect_timeout: Duration::from_millis(300),
        ..EmbedConfig::default()
    });
    let t = std::time::Instant::now();
    assert_eq!(c.embed("a b", true).unwrap().len(), DIM);
    assert!(t.elapsed() < Duration::from_secs(3));
}

#[test]
fn total_failure_is_empty_for_queries_and_error_for_writes() {
    let c = client(vec![DEAD.into()]);
    assert_eq!(c.embed("q", false).unwrap(), Vec::<f32>::new());
    assert!(matches!(
        c.embed("q", true),
        Err(EmbedError::Unavailable { .. })
    ));
    let (z, _) = serve(Reply::Zeros);
    assert!(matches!(
        client(vec![z]).embed("q", true),
        Err(EmbedError::Unavailable { .. })
    ));
}

#[test]
fn wrong_dimension_is_a_typed_error_and_not_retried() {
    let (bad, log) = serve(Reply::WrongDim(768));
    let (good, _) = serve(Reply::Ok);
    match client(vec![bad, good]).embed("q", false) {
        Err(EmbedError::DimensionMismatch {
            expected, actual, ..
        }) => assert_eq!((expected, actual), (DIM, 768)),
        other => panic!("{other:?}"),
    }
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn model_mismatch_is_typed_and_verify_off_lets_it_pass() {
    let (u, _) = serve(Reply::Model("nomic-embed-text"));
    assert!(matches!(
        client(vec![u.clone()]).embed("q", false),
        Err(EmbedError::ModelMismatch { .. })
    ));
    let mut c = client(vec![u]);
    c.config.verify = false;
    assert_eq!(c.embed("q", false).unwrap().len(), DIM);
    let (tagged, _) = serve(Reply::Model("mxbai-embed-large:latest"));
    assert!(client(vec![tagged]).embed("q", false).is_ok());
}

#[test]
fn over_context_400_halves_the_text_and_retries() {
    let (u, log) = serve(Reply::Overflow400);
    let text = "word ".repeat(400);
    let v = client(vec![u]).embed(&text, true).unwrap();
    assert_eq!(v.len(), DIM);
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen, vec![1400, 700, 350, 175]);
}

#[test]
fn input_is_cut_at_1400_chars() {
    let (u, log) = serve(Reply::Ok);
    client(vec![u]).embed(&"é".repeat(5000), true).unwrap();
    assert_eq!(log.lock().unwrap()[0], 1400);
}
