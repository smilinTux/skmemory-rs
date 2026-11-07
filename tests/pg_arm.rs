//! SKMEM-R-04: the pgvector arm against Python's `PGVectorBackend`.
//!
//! `tests/pg_parity/expected.json` was produced by `gen_pg_parity.py` with the
//! real Python backend over `corpus.json` on a throwaway skmem-pg. These tests
//! seed the same corpus and require the same ordered ids for every case.
//!
//! ENV-GATED: the database tests run only when `SKMEM_PG_TEST_DSN` names a
//! loopback Postgres with the `vector` extension (CI provides a service
//! container, with `SKMEM_PG_TEST_REQUIRE=1` so a missing DSN fails instead of
//! skipping); without it they print a SKIP line and pass. They create and
//! drop their own schema, so a plain pgvector image is enough. With
//! `SKMEM_PG_TEST_PUBLIC=1` they also run against `public.memories` of a
//! THROWAWAY skmem-pg (rows prefixed `skrs-r04-` only, deleted afterwards).
#![cfg(feature = "pg")]

mod common;

use common::{serve, Reply};
use postgres::{Client, NoTls};
use serde_json::Value;
use skmem::embed::{EmbedClient, EmbedConfig};
use skmem::model::Layer;
use skmem::pg::{ContextTag, Filters, PgError, VectorArm};
use std::path::Path;

fn read(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/pg_parity")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn test_dsn() -> Option<String> {
    let Some(dsn) = std::env::var("SKMEM_PG_TEST_DSN")
        .ok()
        .filter(|d| !d.is_empty())
    else {
        eprintln!("SKIP: SKMEM_PG_TEST_DSN not set (needs a loopback Postgres with pgvector)");
        return None;
    };
    let cfg: postgres::Config = dsn.parse().expect("SKMEM_PG_TEST_DSN does not parse");
    for h in cfg.get_hosts() {
        if let postgres::config::Host::Tcp(t) = h {
            assert!(
                ["127.0.0.1", "localhost", "::1"].contains(&t.as_str()),
                "SKMEM_PG_TEST_DSN must be loopback"
            );
        }
    }
    Some(dsn)
}

const DDL: &str = "CREATE TABLE {s}.memories (
  id text NOT NULL PRIMARY KEY, layer text, role text, title text, content text, summary text,
  tags text[] DEFAULT '{}'::text[], source text,
  created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
  memory_json jsonb NOT NULL, agent text DEFAULT 'lumina', embedding public.vector(1024)
  {v2});
CREATE INDEX ON {s}.memories USING hnsw (embedding public.vector_cosine_ops);";
const V2: &str = ", kind text, outcome text, event_at timestamptz, valid_from timestamptz, \
                  valid_to timestamptz, supersedes_id text, source_tier text, schema_version integer";

fn seed(client: &mut Client, table: &str, with_v2_cols: bool) {
    let corpus = read("corpus.json");
    for r in corpus.as_array().unwrap() {
        let s = |k: &str| r[k].as_str().map(|x| x.to_string());
        let tags: Vec<String> = r["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let emb = r["has_embedding"].as_bool().unwrap().then(|| {
            pgvector::Vector::from(common::det_embed(&skmem::embed::embed_text(
                r["title"].as_str().unwrap(),
                r["content"].as_str().unwrap(),
                r["summary"].as_str().unwrap(),
            )))
        });
        let mut cols = "id, agent, layer, role, title, content, summary, tags, source, created_at, updated_at, memory_json, embedding".to_string();
        let mut ph =
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10::text::timestamptz,$11::text::timestamptz,$12,$13"
                .to_string();
        let mut p: Vec<Box<dyn postgres::types::ToSql + Sync>> = vec![
            Box::new(s("id")),
            Box::new(s("agent")),
            Box::new(s("layer")),
            Box::new(s("role")),
            Box::new(s("title")),
            Box::new(s("content")),
            Box::new(s("summary").unwrap_or_default()),
            Box::new(tags),
            Box::new(s("source")),
            Box::new(s("created_at")),
            Box::new(s("updated_at")),
            Box::new(r["memory_json"].clone()),
            Box::new(emb),
        ];
        if with_v2_cols && r["v2"].as_bool().unwrap() {
            cols += ", kind, outcome, event_at, valid_from, valid_to, supersedes_id, source_tier, schema_version";
            ph += ",$14,$15,$16::text::timestamptz,$17::text::timestamptz,$18::text::timestamptz,$19,$20,$21";
            p.push(Box::new(s("kind")));
            p.push(Box::new(s("outcome")));
            p.push(Box::new(s("event_at")));
            p.push(Box::new(s("valid_from")));
            p.push(Box::new(s("valid_to")));
            p.push(Box::new(s("supersedes_id")));
            p.push(Box::new(s("source_tier")));
            p.push(Box::new(r["schema_version"].as_i64().map(|v| v as i32)));
        }
        let refs: Vec<&(dyn postgres::types::ToSql + Sync)> =
            p.iter().map(|b| b.as_ref()).collect();
        client
            .execute(
                &format!("INSERT INTO {table} ({cols}) VALUES ({ph})"),
                &refs,
            )
            .unwrap_or_else(|e| panic!("seed {}: {e:?}", r["id"]));
    }
}

fn filters(f: &Value) -> Filters {
    let mut out = Filters::default();
    if let Some(l) = f["layer"].as_str() {
        out.layer = Some(Layer::parse(l).unwrap());
    }
    if let Some(t) = f["tags"].as_array() {
        out.tags = t.iter().map(|x| x.as_str().unwrap().to_string()).collect();
    }
    match &f["context_tag"] {
        Value::String(s) => out.context_tag = Some(ContextTag::One(s.clone())),
        Value::Array(a) => {
            out.context_tag = Some(ContextTag::AnyOf(
                a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            ))
        }
        _ => {}
    }
    out.source = f["source"].as_str().map(|s| s.to_string());
    out.as_of = f["as_of"].as_str().map(|s| s.to_string());
    out
}

fn embedder() -> (Box<EmbedClient>, String) {
    let (url, _) = serve(Reply::Ok);
    (
        Box::new(EmbedClient::new(EmbedConfig {
            urls: vec![url.clone()],
            ..EmbedConfig::default()
        })),
        url,
    )
}

fn arm_on(dsn: &str, search_path: &str, agent: &str) -> VectorArm {
    let mut c = Client::connect(dsn, NoTls).unwrap();
    c.batch_execute(&format!("SET search_path = {search_path}"))
        .unwrap();
    let (e, _) = embedder();
    VectorArm::from_client(c, agent, e).unwrap()
}

fn run_cases(dsn: &str, search_path: &str, only_as_of: bool) -> usize {
    let expected = read("expected.json");
    let mut checked = 0;
    let mut arms: std::collections::HashMap<String, VectorArm> = Default::default();
    for c in expected.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if only_as_of && !name.starts_with("asof_") {
            continue;
        }
        let agent = c["agent"].as_str().unwrap();
        let arm = arms
            .entry(agent.to_string())
            .or_insert_with(|| arm_on(dsn, search_path, agent));
        let got = arm.search_semantic(
            c["query"].as_str().unwrap(),
            c["limit"].as_i64().unwrap(),
            &filters(&c["filters"]),
        );
        if c.get("expect_error").is_some() {
            assert!(matches!(got, Err(PgError::BadAsOf(_))), "{name}: {got:?}");
        } else {
            let hits = got.unwrap_or_else(|e| panic!("{name}: {e}"));
            let dists: Vec<f64> = hits.iter().map(|h| h.distance).collect();
            let want_d: Vec<f64> = c["expect_distances"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(dists.len(), want_d.len(), "case {name} distances");
            for (g, w) in dists.iter().zip(&want_d) {
                assert!((g - w).abs() < 2e-6, "case {name}: distance {g} vs {w}");
            }
            let ids: Vec<String> = hits.into_iter().map(|h| h.memory.id).collect();
            let want: Vec<String> = c["expect_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            assert_eq!(ids, want, "case {name}");
        }
        checked += 1;
    }
    checked
}

/// Create a scratch schema (own DDL, optionally without the v2 typed
/// columns), run `f(schema, search_path, client)`, always drop it.
fn with_schema(dsn: &str, v2: bool, f: impl FnOnce(&str, &str, &mut Client)) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let schema = format!("skrs_r04_{}_{n}", std::process::id());
    let mut c = Client::connect(dsn, NoTls).unwrap();
    // concurrent CREATE EXTENSION races on pg_extension_name_index: serialise it
    c.batch_execute(
        "SELECT pg_advisory_lock(5304); CREATE EXTENSION IF NOT EXISTS vector; \
         SELECT pg_advisory_unlock(5304)",
    )
    .unwrap();
    c.batch_execute(&format!(
        "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}"
    ))
    .unwrap();
    c.batch_execute(
        &DDL.replace("{s}", &schema)
            .replace("{v2}", if v2 { V2 } else { "" }),
    )
    .unwrap();
    let path = format!("{schema}, public");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&schema, &path, &mut c)));
    let _ = c.batch_execute(&format!("DROP SCHEMA {schema} CASCADE"));
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn parity_with_python_typed_columns() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed(c, &format!("{schema}.memories"), true);
        let n = run_cases(&dsn, path, false);
        assert_eq!(n, 46);
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        let _ = arm
            .search_semantic("river stone amber", 10, &Filters::default())
            .unwrap();
        assert!(
            arm.unparseable >= 1,
            "the broken memory_json row must be skipped and counted"
        );
    });
}

#[test]
fn parity_with_python_json_fallback_for_as_of() {
    // No typed columns: validity is read out of memory_json and must give the
    // same answers Python gets from the typed columns.
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, false, |schema, path, c| {
        seed(c, &format!("{schema}.memories"), false);
        let n = run_cases(&dsn, path, true);
        assert!(n >= 10);
        // A garbage valid_to means "still open" (Python's rule): row 003 is
        // closed in the corpus, so it must now appear for a far-future as_of.
        c.batch_execute(&format!(
            "UPDATE {schema}.memories SET memory_json = jsonb_set(memory_json, '{{valid_to}}', '\"garbage\"') \
             WHERE id = 'skrs-r04-003'"
        ))
        .unwrap();
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        let f = Filters {
            as_of: Some("2030-01-01".into()),
            ..Filters::default()
        };
        let ids: Vec<String> = arm
            .search_semantic("river stone amber", 100, &f)
            .unwrap()
            .into_iter()
            .map(|h| h.memory.id)
            .collect();
        assert!(ids.iter().any(|i| i == "skrs-r04-003"), "{ids:?}");
    });
}

#[test]
fn parity_on_public_schema_of_a_throwaway_node() {
    if std::env::var("SKMEM_PG_TEST_PUBLIC").ok().as_deref() != Some("1") {
        eprintln!(
            "SKIP: SKMEM_PG_TEST_PUBLIC != 1 (the public.memories run is for throwaway nodes)"
        );
        return;
    }
    let Some(dsn) = test_dsn() else { return };
    let mut c = Client::connect(&dsn, NoTls).unwrap();
    c.batch_execute("DELETE FROM public.memories WHERE id LIKE 'skrs-r04-%'")
        .unwrap();
    seed(&mut c, "public.memories", true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_eq!(run_cases(&dsn, "public", false), 46);
    }));
    c.batch_execute("DELETE FROM public.memories WHERE id LIKE 'skrs-r04-%'")
        .unwrap();
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn context_tag_empty_list_returns_nothing_not_everything() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed(c, &format!("{schema}.memories"), true);
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        let all = arm
            .search_semantic("river stone", 10, &Filters::default())
            .unwrap();
        assert!(!all.is_empty());
        let f = Filters {
            context_tag: Some(ContextTag::AnyOf(vec![])),
            ..Filters::default()
        };
        assert!(arm
            .search_semantic("river stone", 10, &f)
            .unwrap()
            .is_empty());
    });
}

#[test]
fn dimension_mismatch_is_a_typed_error() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |_, path, _| {
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        match arm.search_vector(vec![0.1; 768], 5, &Filters::default()) {
            Err(PgError::DimensionMismatch { expected, actual }) => {
                assert_eq!((expected, actual), (1024, 768))
            }
            other => panic!("{other:?}"),
        }
        // and an endpoint serving a 768-dim model surfaces the same variant
        let (url, _) = serve(Reply::WrongDim(768));
        let mut c = Client::connect(&dsn, NoTls).unwrap();
        c.batch_execute(&format!("SET search_path = {path}"))
            .unwrap();
        let bad = Box::new(EmbedClient::new(EmbedConfig {
            urls: vec![url],
            ..EmbedConfig::default()
        }));
        let mut arm = VectorArm::from_client(c, "skrs-r04-a", bad).unwrap();
        assert!(matches!(
            arm.search_semantic("q", 5, &Filters::default()),
            Err(PgError::DimensionMismatch {
                expected: 1024,
                actual: 768
            })
        ));
    });
}

#[test]
fn unembeddable_query_abstains_with_empty_result() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |_, path, c| {
        let mut c2 = Client::connect(&dsn, NoTls).unwrap();
        c2.batch_execute(&format!("SET search_path = {path}"))
            .unwrap();
        let _ = c;
        let dead = Box::new(EmbedClient::new(EmbedConfig {
            urls: vec!["http://127.0.0.1:1/api/embed".into()],
            ..EmbedConfig::default()
        }));
        let mut arm = VectorArm::from_client(c2, "skrs-r04-a", dead).unwrap();
        assert!(arm
            .search_semantic("q", 5, &Filters::default())
            .unwrap()
            .is_empty());
    });
}

#[test]
fn dsn_never_appears_in_errors_or_debug() {
    // Hermetic: nothing listens on port 1.
    let pw = ["Sup3r", "SecretPw"].concat();
    let dsn = format!(
        "postgresql://{}:{pw}@{}:1/skrs_db",
        "skrs_user", "127.0.0.1"
    );
    let (e, _) = embedder();
    let err = VectorArm::connect(&dsn, "a", e).unwrap_err();
    let shown = format!("{err} {err:?}");
    for needle in [pw.as_str(), dsn.as_str(), "skrs_user"] {
        assert!(!shown.contains(needle), "{shown}");
    }
    let (e, _) = embedder();
    let bad = VectorArm::connect(&format!("postgresql://u:{pw}@[::1"), "a", e).unwrap_err();
    assert!(!format!("{bad} {bad:?}").contains(pw.as_str()));
}

#[test]
fn remote_dsn_is_refused_by_default_without_connecting() {
    // 192.0.2.1 is TEST-NET-1: nothing is dialled because the check comes first.
    let (e, _) = embedder();
    let pw = ["Sup3r", "SecretPw"].concat();
    let dsn = format!("postgresql://u:{pw}@{}:5432/db", "192.0.2.1");
    match VectorArm::connect_with(&dsn, "a", e, false) {
        Err(PgError::RemoteDsn { host }) => assert_eq!(host, "192.0.2.1"),
        other => panic!("{other:?}"),
    }
    let (e, _) = embedder();
    let err = VectorArm::connect_with(&dsn, "a", e, false).unwrap_err();
    assert!(!format!("{err} {err:?}").contains(pw.as_str()));
    assert!(format!("{err}").contains("SKMEMORY_PG_ALLOW_REMOTE"));
}

#[test]
fn bad_as_of_is_rejected_before_any_sql() {
    // A client that is never used for a query: a bad as_of must fail first.
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |_, path, _| {
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        for bad in [
            "now",
            "yesterday",
            "today",
            "infinity",
            "epoch",
            "2026-13-45",
        ] {
            let f = Filters {
                as_of: Some(bad.into()),
                ..Filters::default()
            };
            assert!(
                matches!(
                    arm.search_semantic("river", 5, &f),
                    Err(PgError::BadAsOf(_))
                ),
                "{bad}"
            );
        }
    });
}

#[test]
fn search_settings_do_not_leak_onto_the_connection() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed(c, &format!("{schema}.memories"), true);
        let mut arm = arm_on(&dsn, path, "skrs-r04-a");
        arm.client_mut()
            .batch_execute("SET TimeZone = 'Asia/Tokyo'; SET hnsw.ef_search = 77")
            .unwrap();
        let hits = arm
            .search_semantic("river stone", 5, &Filters::default())
            .unwrap();
        assert!(!hits.is_empty());
        let show = |arm: &mut VectorArm, what: &str| -> String {
            arm.client_mut()
                .query_one(&format!("SHOW {what}"), &[])
                .unwrap()
                .get(0)
        };
        assert_eq!(show(&mut arm, "TimeZone"), "Asia/Tokyo");
        assert_eq!(show(&mut arm, "hnsw.ef_search"), "77");
        // and the search itself ran with ef_search 200 inside its transaction
        let f = Filters {
            as_of: Some("2026-06-01T00:00:00Z".into()),
            ..Filters::default()
        };
        assert!(arm.search_semantic("river stone", 5, &f).is_ok());
        assert_eq!(show(&mut arm, "TimeZone"), "Asia/Tokyo");
    });
}

#[test]
fn dead_connection_is_re_established() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed(c, &format!("{schema}.memories"), true);
        let mut cfg: postgres::Config = dsn.parse().unwrap();
        cfg.options(&format!("-c search_path={}", path.replace(", ", ",")));
        let (e, _) = embedder();
        let mut arm = VectorArm::connect_config(cfg, "skrs-r04-a", e, false).unwrap();
        let before = arm
            .search_semantic("river stone", 5, &Filters::default())
            .unwrap();
        let pid: i32 = arm
            .client_mut()
            .query_one("SELECT pg_backend_pid()", &[])
            .unwrap()
            .get(0);
        c.execute("SELECT pg_terminate_backend($1)", &[&pid])
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let after = arm
            .search_semantic("river stone", 5, &Filters::default())
            .unwrap();
        let ids = |h: &[skmem::pg::Hit]| h.iter().map(|x| x.memory.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&before), ids(&after));
        // an arm wrapped around a caller's client cannot reconnect and says so
        let mut c2 = Client::connect(&dsn, NoTls).unwrap();
        c2.batch_execute(&format!("SET search_path = {path}"))
            .unwrap();
        let pid2: i32 = c2.query_one("SELECT pg_backend_pid()", &[]).unwrap().get(0);
        let (e, _) = embedder();
        let mut wrapped = VectorArm::from_client(c2, "skrs-r04-a", e).unwrap();
        c.execute("SELECT pg_terminate_backend($1)", &[&pid2])
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(wrapped
            .search_semantic("river", 5, &Filters::default())
            .is_err());
    });
}
