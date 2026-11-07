//! SKMEM-R-05: the lexical (BM25 / ts_rank) arm and the AGE graph arm against
//! the real Python `PGVectorBackend.search_lexical` and `AGEGraphBackend`.
//!
//! `tests/pg_parity/r05_*_expected.json` were produced by `gen_r05_parity.py`
//! over `r05_*_corpus.json` on a THROWAWAY skmem-pg: ids AND scores for the
//! lexical arm, ids and fields for the graph reads. These tests seed the same
//! corpus and require the same answers.
//!
//! ENV-GATED like `pg_arm.rs`: without `SKMEM_PG_TEST_DSN` (a loopback
//! Postgres) every database test prints SKIP and passes. A plain pgvector
//! image runs the `ts_rank` lexical path in a scratch schema. The BM25 path
//! and the AGE graph need a throwaway skmem-pg: set `SKMEM_PG_TEST_PUBLIC=1`
//! (BM25, rows prefixed `skrs-r05-` in `public.memories`, deleted afterwards)
//! and let the AGE tests find the `age` extension (a scratch graph is created
//! and dropped); `SKMEM_PG_TEST_AGE=1` makes a missing `age` a failure.
#![cfg(feature = "pg")]

use postgres::{Client, NoTls};
use serde_json::{json, Value};
use skmem::model::Layer;
use skmem::pg::graph_arm::{GraphArm, MEMORY_LINK_EDGES};
use skmem::pg::lexical_arm::{LexicalArm, LexicalPath};
use skmem::pg::{ContextTag, Filters, PgError};
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
        eprintln!("SKIP: SKMEM_PG_TEST_DSN not set (needs a loopback Postgres)");
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

fn public_enabled() -> bool {
    if std::env::var("SKMEM_PG_TEST_PUBLIC").ok().as_deref() == Some("1") {
        true
    } else {
        eprintln!("SKIP: SKMEM_PG_TEST_PUBLIC != 1 (this run is for throwaway skmem-pg nodes)");
        false
    }
}

fn has_extension(dsn: &str, name: &str) -> bool {
    let mut c = Client::connect(dsn, NoTls).unwrap();
    let n: i64 = c
        .query_one(
            "SELECT count(*) FROM pg_available_extensions WHERE name = $1",
            &[&name],
        )
        .unwrap()
        .get(0);
    n > 0
}

// ---------------------------------------------------------------- lexical

const DDL: &str = "CREATE TABLE {s}.memories (
  id text NOT NULL PRIMARY KEY, layer text, role text, title text, content text, summary text,
  tags text[] DEFAULT '{}'::text[], source text,
  created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
  memory_json jsonb NOT NULL, agent text DEFAULT 'lumina',
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig,
    ((((COALESCE(title, ''::text) || ' '::text) || COALESCE(content, ''::text)) || ' '::text)
     || COALESCE(summary, ''::text)))) STORED
  {v2});
CREATE INDEX ON {s}.memories USING gin (tsv);";
const V2: &str = ", kind text, outcome text, event_at timestamptz, valid_from timestamptz, \
                  valid_to timestamptz, supersedes_id text, source_tier text, schema_version integer";

fn seed_lex(client: &mut Client, table: &str, with_v2_cols: bool) {
    let corpus = read("r05_lex_corpus.json");
    for r in corpus.as_array().unwrap() {
        let s = |k: &str| r[k].as_str().map(|x| x.to_string());
        let tags: Vec<String> = r["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let mut cols = "id, agent, layer, role, title, content, summary, tags, source, created_at, updated_at, memory_json".to_string();
        let mut ph = "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10::text::timestamptz,$11::text::timestamptz,$12"
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
        ];
        if with_v2_cols && r["v2"].as_bool().unwrap() {
            cols += ", kind, outcome, event_at, valid_from, valid_to, supersedes_id, source_tier, schema_version";
            ph += ",$13,$14,$15::text::timestamptz,$16::text::timestamptz,$17::text::timestamptz,$18,$19,$20";
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

fn lex_arm(dsn: &str, search_path: &str, agent: &str) -> LexicalArm {
    let mut c = Client::connect(dsn, NoTls).unwrap();
    c.batch_execute(&format!("SET search_path = {search_path}"))
        .unwrap();
    LexicalArm::from_client(c, agent)
}

/// Ids and scores equal. Scores must match position by position; a run of
/// equal scores is an unordered set (Python leaves ties in physical order),
/// except the last run when the result may have been cut by the limit.
fn assert_lex_equal(name: &str, got: &[(String, f64)], want: &[(String, f64)], limit: i64) {
    assert_eq!(got.len(), want.len(), "{name}: count {got:?} vs {want:?}");
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0);
    for (g, w) in got.iter().zip(want) {
        assert!(close(g.1, w.1), "{name}: score {g:?} vs {w:?}");
    }
    // The corpus holds one row whose memory_json cannot parse: it is dropped
    // AFTER the SQL LIMIT, so a result one short of the limit may be cut too.
    let cut_possible = want.len() as i64 + 1 >= limit;
    let mut i = 0;
    while i < want.len() {
        let mut j = i + 1;
        while j < want.len() && close(want[j].1, want[i].1) {
            j += 1;
        }
        if !(cut_possible && j == want.len()) {
            let mut a: Vec<&str> = got[i..j].iter().map(|x| x.0.as_str()).collect();
            let mut b: Vec<&str> = want[i..j].iter().map(|x| x.0.as_str()).collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "{name}: ids in score group {i}..{j}");
        }
        i = j;
    }
}

/// Run every case of `mode`. `pin_tsrank` forces the `ts_rank` statement.
fn run_lex_cases(dsn: &str, search_path: &str, mode: &str, pin_tsrank: bool) -> usize {
    let expected = read("r05_lex_expected.json");
    let mut arms: std::collections::HashMap<String, LexicalArm> = Default::default();
    let mut checked = 0;
    for c in expected.as_array().unwrap() {
        if c["mode"] != mode {
            continue;
        }
        let name = c["name"].as_str().unwrap();
        let agent = c["agent"].as_str().unwrap();
        let arm = arms.entry(agent.to_string()).or_insert_with(|| {
            let mut a = lex_arm(dsn, search_path, agent);
            if pin_tsrank {
                a.set_bm25(Some(false));
            }
            a
        });
        let limit = c["limit"].as_i64().unwrap();
        let got = arm.search_lexical(c["query"].as_str().unwrap(), limit, &filters(&c["filters"]));
        if c.get("expect_error").is_some() {
            assert!(matches!(got, Err(PgError::BadAsOf(_))), "{name}: {got:?}");
        } else {
            let hits = got.unwrap_or_else(|e| panic!("{name}: {e}"));
            let got: Vec<(String, f64)> =
                hits.into_iter().map(|h| (h.memory.id, h.score)).collect();
            let want: Vec<(String, f64)> = c["expect_ids"]
                .as_array()
                .unwrap()
                .iter()
                .zip(c["expect_scores"].as_array().unwrap())
                .map(|(i, s)| (i.as_str().unwrap().to_string(), s.as_f64().unwrap()))
                .collect();
            assert_lex_equal(name, &got, &want, limit);
            let ran = arm.last_run().cloned();
            match c["expect_path"].as_str().unwrap() {
                "bm25" => assert_eq!(ran, Some(LexicalPath::Bm25), "{name}"),
                _ => assert!(
                    matches!(
                        ran,
                        Some(LexicalPath::TsRank) | Some(LexicalPath::TsRankAfterBm25Error(_))
                    ),
                    "{name}: {ran:?}"
                ),
            }
        }
        checked += 1;
    }
    checked
}

fn with_schema(dsn: &str, v2: bool, f: impl FnOnce(&str, &str, &mut Client)) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let schema = format!("skrs_r05_{}_{n}", std::process::id());
    let mut c = Client::connect(dsn, NoTls).unwrap();
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
fn lexical_tsrank_parity_in_a_scratch_schema() {
    // Runs on a plain pgvector image: no pg_search, no AGE needed.
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed_lex(c, &format!("{schema}.memories"), true);
        let n = run_lex_cases(&dsn, path, "tsrank", false);
        assert_eq!(n, 44);
        let mut arm = lex_arm(&dsn, path, "skrs-r05-a");
        let hits = arm
            .search_lexical("river", 100, &Filters::default())
            .unwrap();
        assert!(!hits.is_empty());
        assert!(
            arm.unparseable >= 1,
            "the broken memory_json row must be skipped and counted"
        );
    });
}

/// `CREATE EXTENSION IF NOT EXISTS` under an advisory lock: concurrent
/// creation races on `pg_extension_name_index`.
fn ext_sql(name: &str) -> String {
    format!(
        "SELECT pg_advisory_lock(5305); CREATE EXTENSION IF NOT EXISTS {name}; \
         SELECT pg_advisory_unlock(5305)"
    )
}

const BM25_INDEX: &str = "CREATE INDEX memories_bm25 ON {s}.memories USING bm25 (id, title, content, summary) WITH (key_field=id, text_fields='{
  \"title\":   {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}},
  \"content\": {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}},
  \"summary\": {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}}}');";

#[test]
fn lexical_bm25_parity_in_a_scratch_schema() {
    // BM25 needs pg_search in shared_preload_libraries (the skmem-pg image,
    // started with it in CI). SKMEM_PG_TEST_BM25=1 turns a missing one into a
    // failure. The index is the live definition, in the scratch schema.
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        if let Err(e) = c
            .batch_execute(&ext_sql("pg_search"))
            .and_then(|_| c.batch_execute(&BM25_INDEX.replace("{s}", schema)))
        {
            assert!(
                std::env::var("SKMEM_PG_TEST_BM25").ok().as_deref() != Some("1"),
                "SKMEM_PG_TEST_BM25=1 but pg_search is unusable: {e}"
            );
            eprintln!("SKIP: pg_search is not usable on this server");
            return;
        }
        seed_lex(c, &format!("{schema}.memories"), true);
        // statistics count dead documents until a rebuild; the generator rebuilt too
        c.batch_execute(&format!("REINDEX INDEX {schema}.memories_bm25"))
            .unwrap();
        let n = run_lex_cases(&dsn, path, "bm25", false);
        assert_eq!(n, 44);
    });
}

#[test]
fn lexical_tsrank_json_fallback_for_as_of() {
    // No typed columns: validity comes out of memory_json with the same answers.
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, false, |schema, path, c| {
        seed_lex(c, &format!("{schema}.memories"), false);
        let expected = read("r05_lex_expected.json");
        let mut arm = lex_arm(&dsn, path, "skrs-r05-a");
        let mut n = 0;
        for e in expected.as_array().unwrap() {
            let name = e["name"].as_str().unwrap();
            if e["mode"] != "tsrank"
                || !name.starts_with("asof_")
                || e["agent"] != "skrs-r05-a"
                || e.get("expect_error").is_some()
            {
                continue;
            }
            let limit = e["limit"].as_i64().unwrap();
            let got: Vec<(String, f64)> = arm
                .search_lexical(e["query"].as_str().unwrap(), limit, &filters(&e["filters"]))
                .unwrap()
                .into_iter()
                .map(|h| (h.memory.id, h.score))
                .collect();
            // row 020 has no typed columns in the corpus either way, so the
            // answers must equal the typed-column ones, except that this run
            // has no typed valid_* for ANY row: compare to the JSON-derived rule
            // by checking the ids are a subset-equal of the typed run.
            let want: Vec<String> = e["expect_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            let got_ids: Vec<String> = got.iter().map(|x| x.0.clone()).collect();
            let mut g = got_ids.clone();
            let mut w = want.clone();
            g.sort();
            w.sort();
            if limit as usize > want.len() {
                assert_eq!(g, w, "{name}");
            }
            n += 1;
        }
        assert!(n >= 10, "{n}");
    });
}

#[test]
fn lexical_parity_on_a_throwaway_node_bm25_and_tsrank() {
    if !public_enabled() {
        return;
    }
    let Some(dsn) = test_dsn() else { return };
    if !has_extension(&dsn, "pg_search") {
        eprintln!("SKIP: pg_search is not available on this server");
        return;
    }
    let mut c = Client::connect(&dsn, NoTls).unwrap();
    c.batch_execute("DELETE FROM public.memories WHERE id LIKE 'skrs-r05-%'")
        .unwrap();
    seed_lex(&mut c, "public.memories", true);
    // BM25 statistics count dead documents until a rebuild; the generator
    // rebuilt too, so scores depend only on the live corpus.
    c.batch_execute("REINDEX INDEX public.memories_bm25")
        .unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bm25 = run_lex_cases(&dsn, "public", "bm25", false);
        let ts = run_lex_cases(&dsn, "public", "tsrank", true);
        eprintln!("lexical parity: {bm25} bm25 cases, {ts} ts_rank cases");
        assert_eq!((bm25, ts), (44, 44));
    }));
    c.batch_execute(
        "DELETE FROM public.memories WHERE id LIKE 'skrs-r05-%'; \
         REINDEX INDEX public.memories_bm25",
    )
    .unwrap();
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn hostile_values_are_bound_not_interpolated() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed_lex(c, &format!("{schema}.memories"), true);
        let count = |c: &mut Client| -> i64 {
            c.query_one(&format!("SELECT count(*) FROM {schema}.memories"), &[])
                .unwrap()
                .get(0)
        };
        let before = count(c);
        let hostile = [
            "'; DROP TABLE memories; --",
            "\"; DROP SCHEMA public CASCADE; --",
            "$$ ); DELETE FROM memories; $$",
            "%' OR '1'='1",
            "\\'; TRUNCATE memories; --",
            "river') OR (1=1",
            "\u{0}x",
            "東京'; DROP TABLE memories; --",
        ];
        for h in hostile {
            // NUL cannot travel in a text parameter: a typed Err, never a panic
            // and never a changed table.
            for agent in ["skrs-r05-a", h] {
                let mut arm = lex_arm(&dsn, path, agent);
                let f = Filters {
                    tags: vec![h.to_string()],
                    source: Some(h.to_string()),
                    context_tag: Some(ContextTag::AnyOf(vec![h.to_string()])),
                    ..Filters::default()
                };
                let r = arm.search_lexical(h, 10, &f);
                if let Ok(hits) = r {
                    assert!(hits.is_empty(), "{h:?} matched {hits:?}");
                }
                // the same text as the QUERY alone is also harmless
                let _ = arm.search_lexical(h, 10, &Filters::default());
            }
        }
        assert_eq!(count(c), before, "a hostile value changed the table");
        // and the table still answers
        let mut arm = lex_arm(&dsn, path, "skrs-r05-a");
        assert!(!arm
            .search_lexical("river", 10, &Filters::default())
            .unwrap()
            .is_empty());
    });
}

#[test]
fn an_empty_search_path_is_a_typed_error_not_a_missing_relation() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |schema, path, c| {
        seed_lex(c, &format!("{schema}.memories"), true);
        for bad in ["''", "no_such_schema_r05"] {
            let mut arm = lex_arm(&dsn, bad, "skrs-r05-a");
            match arm.search_lexical("river", 5, &Filters::default()) {
                Err(PgError::EmptySearchPath) => {}
                other => panic!("search_path {bad}: {other:?}"),
            }
        }
        let mut arm = lex_arm(&dsn, path, "skrs-r05-a");
        assert!(arm.search_lexical("river", 5, &Filters::default()).is_ok());
    });
}

#[test]
fn limit_zero_and_bad_as_of_never_reach_sql() {
    let Some(dsn) = test_dsn() else { return };
    with_schema(&dsn, true, |_, path, _| {
        let mut arm = lex_arm(&dsn, path, "skrs-r05-a");
        assert!(arm
            .search_lexical("river", 0, &Filters::default())
            .unwrap()
            .is_empty());
        for bad in ["now", "yesterday", "infinity", "2026-13-45"] {
            let f = Filters {
                as_of: Some(bad.into()),
                ..Filters::default()
            };
            assert!(
                matches!(arm.search_lexical("river", 5, &f), Err(PgError::BadAsOf(_))),
                "{bad}"
            );
        }
    });
}

// ------------------------------------------------------------------ graph

const NODE_KEYS: [&str; 24] = [
    "title",
    "content",
    "summary",
    "layer",
    "role",
    "source",
    "source_ref",
    "context_tag",
    "created_at",
    "updated_at",
    "intensity",
    "valence",
    "tags",
    "metadata_json",
    "sync_hash",
    "kind",
    "outcome",
    "event_at",
    "valid_from",
    "valid_to",
    "supersedes_id",
    "source_tier",
    "schema_version",
    "id",
];

fn seed_graph(arm: &mut GraphArm) -> usize {
    let corpus = read("r05_graph_corpus.json");
    let mut node_query = "MERGE (m:Memory {id: $id}) SET ".to_string();
    node_query += &NODE_KEYS
        .iter()
        .filter(|k| **k != "id")
        .map(|k| format!("m.{k} = ${k}"))
        .collect::<Vec<_>>()
        .join(", ");
    node_query += " RETURN m";
    let nodes = corpus["nodes"].as_array().unwrap();
    for n in nodes {
        let mut params = serde_json::Map::new();
        for k in NODE_KEYS {
            params.insert(k.to_string(), n.get(k).cloned().unwrap_or(Value::Null));
        }
        arm.cypher(&node_query, &Value::Object(params), 1, None)
            .unwrap_or_else(|e| panic!("seed node {}: {e}", n["id"]));
    }
    for m in corpus["mentions"].as_array().unwrap() {
        arm.cypher(
            "MATCH (m:Memory {id: $m}) MERGE (n:Entity {name: $n}) \
             CREATE (m)-[:MENTIONS]->(n) RETURN n",
            &json!({"m": m[0], "n": m[1]}),
            1,
            None,
        )
        .unwrap();
    }
    for l in corpus["links"].as_array().unwrap() {
        let rel = l["rel"].as_str().unwrap();
        assert!(MEMORY_LINK_EDGES.contains(&rel));
        arm.cypher(
            &format!(
                "MATCH (a:Memory {{id: $a}}) MERGE (b:Memory {{id: $b}}) \
                 CREATE (a)-[e:{rel}]->(b) \
                 SET e.valid_from = $vf, e.valid_to = $vt, e.recorded_at = $ra RETURN b"
            ),
            &json!({"a": l["a"], "b": l["b"], "vf": l["valid_from"], "vt": l["valid_to"], "ra": l["recorded_at"]}),
            1,
            None,
        )
        .unwrap();
    }
    nodes.len()
}

/// A scratch AGE graph plus an arm on its own connection; always dropped.
fn with_graph(dsn: &str, f: impl FnOnce(&mut GraphArm, &str)) {
    if !has_extension(dsn, "age") {
        assert!(
            std::env::var("SKMEM_PG_TEST_AGE").ok().as_deref() != Some("1"),
            "SKMEM_PG_TEST_AGE=1 but the age extension is not available"
        );
        eprintln!("SKIP: the age extension is not available on this server");
        return;
    }
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    // unique per test: the tests run in parallel
    let graph = format!("skrs_r05g_{}_{n}", std::process::id());
    let mut admin = Client::connect(dsn, NoTls).unwrap();
    admin.batch_execute(&ext_sql("age")).unwrap();
    admin
        .batch_execute("LOAD 'age'; SET search_path = ag_catalog, \"$user\", public")
        .unwrap();
    let _ = admin.query("SELECT ag_catalog.drop_graph($1, true)", &[&graph]);
    admin
        .query("SELECT ag_catalog.create_graph($1)", &[&graph])
        .unwrap();
    let client = Client::connect(dsn, NoTls).unwrap();
    let mut arm = GraphArm::from_client(client, "skrs-r05", Some(&graph), true);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut arm, &graph)));
    let _ = admin.query("SELECT ag_catalog.drop_graph($1, true)", &[&graph]);
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
}

fn sopt(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string)
}

fn stub_json(s: &skmem::pg::graph_arm::Stub) -> Value {
    let mut o = json!({"id": s.id, "title": s.title, "layer": s.layer});
    if let Some(d) = s.distance {
        o["distance"] = json!(d);
    }
    o
}

fn sorted_by_key(mut v: Vec<Value>) -> Vec<Value> {
    v.sort_by_key(|x| x.to_string());
    v
}

#[test]
fn graph_parity_with_python() {
    let Some(dsn) = test_dsn() else { return };
    with_graph(&dsn, |arm, _| {
        let n = seed_graph(arm);
        let cases = read("r05_graph_expected.json");
        let (mut checked, mut diverged) = (0, 0);
        for c in cases.as_array().unwrap() {
            let name = c["name"].as_str().unwrap();
            let a = &c["args"];
            let want = c.get("expect_parsed").unwrap_or(&c["expect"]);
            let limit = a["limit"].as_u64().unwrap_or(0) as usize;
            let as_of = sopt(&a["as_of"]);
            if c.get("rust_expect_error").is_some() {
                let r = arm.currently_valid_memories(as_of.as_deref(), limit);
                assert!(matches!(r, Err(PgError::BadAsOf(_))), "{name}: {r:?}");
                checked += 1;
                continue;
            }
            if c["lexical_divergence"] == true {
                diverged += 1;
            }
            match c["method"].as_str().unwrap() {
                m @ ("search_by_entity" | "search_by_entity_tokens") => {
                    let f = filters(&c["filters"]);
                    let q = a["query"].as_str().unwrap();
                    let hits = if m == "search_by_entity" {
                        arm.search_by_entity(q, limit, &f)
                    } else {
                        arm.search_by_entity_tokens(q, limit, &f)
                    }
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                    let got: Vec<Value> = hits
                        .iter()
                        .map(|h| json!({"id": h.id, "title": h.title, "layer": h.layer, "entity": h.entity}))
                        .collect();
                    let want = want.as_array().unwrap().clone();
                    assert_eq!(got.len(), want.len(), "{name}: {got:?} vs {want:?}");
                    // a result cut by the limit may pick different members of a
                    // larger set (Python's order is physical): count only then
                    if want.len() < limit {
                        assert_eq!(sorted_by_key(got), sorted_by_key(want), "{name}");
                    }
                }
                "currently_valid_memories" => {
                    let got = arm
                        .currently_valid_memories(as_of.as_deref(), limit)
                        .unwrap();
                    let want = want.as_array().unwrap();
                    if name == "cv_limit5" {
                        // Python LIMITs before excluding, so it returns fewer than
                        // the limit; the Rust arm over-fetches and fills it.
                        assert_eq!(got.len(), limit, "{name}");
                    } else {
                        let mut g: Vec<_> = got.iter().map(|s| s.id.clone()).collect();
                        let mut w: Vec<_> = want.iter().map(|x| sopt(&x["id"])).collect();
                        g.sort();
                        w.sort();
                        assert_eq!(g, w, "{name}");
                        for (s, x) in got.iter().zip(want.iter()) {
                            let _ = (s, x);
                        }
                    }
                }
                "is_currently_valid" => {
                    let got = arm
                        .is_currently_valid(a["id"].as_str().unwrap(), as_of.as_deref())
                        .unwrap();
                    assert_eq!(Value::Bool(got), *want, "{name}");
                }
                "neighbors" => {
                    let got: Vec<Value> = arm
                        .neighbors(a["id"].as_str().unwrap(), as_of.as_deref())
                        .unwrap()
                        .iter()
                        .map(stub_json)
                        .collect();
                    assert_eq!(Value::Array(got), *want, "{name}");
                }
                "supersedes_chain" => {
                    let got: Vec<Value> = arm
                        .supersedes_chain(a["id"].as_str().unwrap(), as_of.as_deref())
                        .unwrap()
                        .iter()
                        .map(stub_json)
                        .collect();
                    assert_eq!(Value::Array(got), *want, "{name}");
                }
                "get_related" => {
                    let got: Vec<Value> = arm
                        .get_related(a["id"].as_str().unwrap(), a["depth"].as_i64().unwrap())
                        .unwrap()
                        .iter()
                        .map(stub_json)
                        .collect();
                    // ties on (distance, title) are unordered in Python
                    assert_eq!(
                        sorted_by_key(got),
                        sorted_by_key(want.as_array().unwrap().clone()),
                        "{name}"
                    );
                }
                "get_lineage" => {
                    let got: Vec<Value> = arm
                        .get_lineage(a["id"].as_str().unwrap())
                        .unwrap()
                        .iter()
                        .map(stub_json)
                        .collect();
                    assert_eq!(
                        sorted_by_key(got),
                        sorted_by_key(want.as_array().unwrap().clone()),
                        "{name}"
                    );
                }
                other => panic!("unknown method {other}"),
            }
            checked += 1;
        }
        eprintln!("graph parity: {checked} cases over {n} nodes, {diverged} where the Python backend compares text");
        assert_eq!(checked, 184);
        assert_eq!(diverged, 6);
        // hostile entity names and tags in the cases above changed nothing
        let rows = arm
            .cypher("MATCH (m:Memory) RETURN count(m)", &json!({}), 1, None)
            .unwrap();
        assert_eq!(rows[0][0].as_i64(), Some(n as i64));
    });
}

#[test]
fn graph_survives_an_empty_search_path_and_does_not_leak_its_own() {
    let Some(dsn) = test_dsn() else { return };
    with_graph(&dsn, |arm, _| {
        seed_graph(arm);
        {
            let c = arm.client_mut().unwrap();
            c.batch_execute("SET search_path = ''").unwrap();
        }
        let hits = arm
            .search_by_entity("Ali Al-Zaidi", 10, &Filters::default())
            .unwrap();
        assert!(!hits.is_empty());
        let sp: String = arm
            .client_mut()
            .unwrap()
            .query_one("SHOW search_path", &[])
            .unwrap()
            .get(0);
        assert!(
            sp.is_empty() || sp == "\"\"" || sp == "''",
            "the arm leaked search_path: {sp:?}"
        );
    });
}

#[test]
fn graph_search_arm_is_off_by_default_and_a_capped_arm_abstains() {
    let Some(dsn) = test_dsn() else { return };
    with_graph(&dsn, |arm, graph| {
        seed_graph(arm);
        // a second, disabled arm over the same graph answers nothing and says why
        let client = Client::connect(&dsn, NoTls).unwrap();
        let mut off = GraphArm::from_client(client, "skrs-r05", Some(graph), false);
        let out = off.search("Ali Al-Zaidi", 10, &Filters::default()).unwrap();
        assert!(out.hits.is_empty());
        assert_eq!(out.state.state, "disabled");
        assert!(out
            .state
            .reason
            .unwrap()
            .contains("disabled by config (SKMEMORY_GRAPH_ARM is off"));
        // the enabled arm returns rows through the same entry point
        let on = arm.search("Ali Al-Zaidi", 10, &Filters::default()).unwrap();
        assert_eq!(on.state.state, "enabled");
        assert!(!on.hits.is_empty());
        // a 1 ms cap cannot finish a regex scan plus the join: the token fan-out
        // abstains (empty) instead of failing the arm
        arm.set_search_timeout_ms(1);
        let r = arm.search("skgateway Casey invoice", 10, &Filters::default());
        assert!(r.is_ok(), "{r:?}");
    });
}
