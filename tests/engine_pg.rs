//! SKMEM-R-07C: the class bench through `skmem::engine::Engine` with the
//! Postgres arms (pgvector `semantic`, optionally BM25 `lexical`) over a seeded
//! scratch schema on a THROWAWAY skmem-pg, plus the connection guarantees.
//!
//! ENV-GATED like the other pg tests: `SKMEM_PG_TEST_DSN` (loopback only),
//! `SKMEM_PG_TEST_REQUIRE=1` makes a missing DSN a failure, `SKMEM_PG_TEST_BM25=1`
//! makes an unusable pg_search a failure.
#![cfg(feature = "pg")]

use postgres::{Client, NoTls};
use serde_json::Value;
use skmem::bench::{self, hashed_dense, EngineArms, HashedEmbedder};
use skmem::engine::pg_arms::{attach, PgConfig};
use skmem::model::Memory;
use std::path::Path;
use std::sync::Arc;

fn dsn() -> Option<String> {
    let Some(dsn) = std::env::var("SKMEM_PG_TEST_DSN")
        .ok()
        .filter(|d| !d.is_empty())
    else {
        assert!(
            std::env::var("SKMEM_PG_TEST_REQUIRE").ok().as_deref() != Some("1"),
            "SKMEM_PG_TEST_REQUIRE=1 but SKMEM_PG_TEST_DSN is not set"
        );
        eprintln!("SKIP: SKMEM_PG_TEST_DSN not set");
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
  memory_json jsonb NOT NULL, agent text DEFAULT 'lumina', embedding public.vector(1024),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig,
    ((((COALESCE(title, ''::text) || ' '::text) || COALESCE(content, ''::text)) || ' '::text)
     || COALESCE(summary, ''::text)))) STORED,
  kind text, outcome text, event_at timestamptz, valid_from timestamptz, valid_to timestamptz,
  supersedes_id text, source_tier text, schema_version integer);
CREATE INDEX ON {s}.memories USING hnsw (embedding public.vector_cosine_ops);
CREATE INDEX ON {s}.memories USING gin (tsv);";

const BM25_INDEX: &str = "CREATE INDEX memories_bm25 ON {s}.memories USING bm25 (id, title, content, summary) WITH (key_field=id, text_fields='{
  \"title\":   {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}},
  \"content\": {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}},
  \"summary\": {\"tokenizer\": {\"type\": \"default\", \"stemmer\": \"English\", \"stopwords_language\": \"English\"}}}');";

const AGENT: &str = "skrs-r07c";

fn seed(c: &mut Client, schema: &str, mems: &[Memory]) {
    for m in mems {
        let tags: Vec<String> = m.tags.clone();
        let json: Value = m.to_value();
        let emb = pgvector::Vector::from(hashed_dense(&bench::doc_text(m)));
        c.execute(
            &format!(
                "INSERT INTO {schema}.memories (id, agent, layer, role, title, content, summary, tags, source, \
                 created_at, updated_at, memory_json, embedding, kind, outcome, event_at, valid_from, valid_to, \
                 supersedes_id, source_tier, schema_version) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,\
                 $10::text::timestamptz,$11::text::timestamptz,$12,$13,$14,$15,$16::text::timestamptz,\
                 $17::text::timestamptz,$18::text::timestamptz,$19,$20,$21)"
            ),
            &[
                &m.id,
                &AGENT,
                &m.layer.as_str(),
                &m.role.as_str(),
                &m.title,
                &m.content,
                &m.summary,
                &tags,
                &m.source,
                &m.created_at,
                &m.updated_at,
                &json,
                &emb,
                &m.kind,
                &m.outcome,
                &m.event_at,
                &m.valid_from,
                &m.valid_to,
                &m.supersedes_id,
                &m.source_tier,
                &(m.schema_version as i32),
            ],
        )
        .unwrap_or_else(|e| panic!("seed {}: {e:?}", m.id));
    }
}

fn with_schema(dsn: &str, bm25: bool, f: impl FnOnce(&str, &PgConfigFactory)) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let schema = format!("skrs_r07c_{}_{n}", std::process::id());
    let mut c = Client::connect(dsn, NoTls).unwrap();
    c.batch_execute(&format!(
        "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}"
    ))
    .unwrap();
    c.batch_execute(&DDL.replace("{s}", &schema)).unwrap();
    if bm25 {
        c.batch_execute(
            "SELECT pg_advisory_xact_lock(5305); CREATE EXTENSION IF NOT EXISTS pg_search",
        )
        .and_then(|_| c.batch_execute(&BM25_INDEX.replace("{s}", &schema)))
        .expect("pg_search is required for the BM25 arm (use the skmem-pg image)");
    }
    let factory = PgConfigFactory {
        dsn: dsn.to_string(),
        path: format!("{schema},public"),
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f(&schema, &factory);
    }));
    let _ = c.batch_execute(&format!("DROP SCHEMA {schema} CASCADE"));
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
}

struct PgConfigFactory {
    dsn: String,
    path: String,
}

impl PgConfigFactory {
    fn config(&self, lexical: bool, read_only: bool) -> PgConfig {
        let mut c = PgConfig::new(self.dsn.clone(), read_only, false);
        c.embedder = Arc::new(|| Box::new(HashedEmbedder));
        c.options = Some(format!("-c search_path={}", self.path));
        c.lexical = lexical;
        c.graph = false;
        c
    }
}

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval-v1")
}

fn golden() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("class-bench-python.json")).unwrap())
        .unwrap()
}

/// Test harness only: keep pgvector hits whose hashed cosine to the query is
/// above zero and order them by the bench's own rule (cosine descending, ties by
/// id). Python's bench arm is an in-memory cosine with that floor and tie-break,
/// while pgvector's KNN has neither (it fills `LIMIT` with the nearest rows
/// however far, and breaks f32 ties arbitrarily). With both applied the Postgres
/// plumbing (SQL, filters, scope, hydration) must give the same page as the
/// golden; without them the tails and tie order legitimately differ.
struct CosineFloor(Box<dyn skmem::engine::MemoryArm>);

impl skmem::engine::MemoryArm for CosineFloor {
    fn search(
        &mut self,
        query: &str,
        limit: usize,
        f: &skmem::engine::Filters,
        scope: &skmem::audience::ReaderScope,
    ) -> Result<Vec<Memory>, String> {
        let qv = bench::concept_vector(query);
        let hits = self.0.search(query, limit, f, scope)?;
        let mut scored: Vec<(f64, Memory)> = hits
            .into_iter()
            .map(|m| {
                let dv = bench::concept_vector(&bench::doc_text(&m));
                (qv.iter().zip(&dv).map(|(a, b)| a * b).sum::<f64>(), m)
            })
            .filter(|(c, _)| *c > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
        Ok(scored.into_iter().map(|(_, m)| m).collect())
    }
}

fn run(dsn: &str, lexical: bool, floor: bool) -> bench::BenchResult {
    with_schema_result(dsn, lexical, floor)
}

fn with_schema_result(dsn: &str, lexical: bool, floor: bool) -> bench::BenchResult {
    let mut out = None;
    with_schema(dsn, lexical, |schema, fac| {
        let schema = schema.to_string();
        let dsn2 = fac.dsn.clone();
        let cfg = fac.config(lexical, false);
        let res = bench::run_bench_engine(
            &dir(),
            EngineArms::Custom(Box::new(move |mems: &[Memory]| {
                let mut c = Client::connect(&dsn2, NoTls).map_err(|e| e.to_string())?;
                seed(&mut c, &schema, mems);
                if lexical {
                    c.batch_execute(&format!("REINDEX INDEX {schema}.memories_bm25"))
                        .map_err(|e| e.to_string())?;
                }
                let mut arms = attach(&cfg, AGENT);
                if floor {
                    arms.semantic = skmem::engine::Attach::ready(Box::new(CosineFloor(
                        skmem::engine::pg_arms::vector_arm(&cfg, AGENT),
                    )));
                }
                Ok(arms)
            })),
        )
        .expect("engine bench over pg");
        out = Some(res);
    });
    out.unwrap()
}

fn page_diffs(res: &bench::BenchResult) -> Vec<String> {
    let g = golden();
    let mut diff = Vec::new();
    for row in &res.rows {
        let want: Vec<&str> = g["candidate_pages"][&row.id]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let got: Vec<&str> = row.candidate_page.iter().map(String::as_str).collect();
        if got != want {
            diff.push(format!("{}: rust {got:?} python {want:?}", row.id));
        }
    }
    diff
}

fn worse(res: &bench::BenchResult) -> Vec<String> {
    let g = golden();
    let mut worse = Vec::new();
    for (class, entry) in g["by_class"].as_object().unwrap() {
        let py = &entry["strategies"]["candidate"];
        let (_, rust) = &res.by_class[class];
        let rs = &rust["candidate"];
        for (metric, got) in [
            ("recall@5", rs.recall_at_5),
            ("mrr", rs.mrr),
            ("abstain_correct", rs.abstain_correct),
        ] {
            if let (Some(want), Some(got)) = (py[metric].as_f64(), got) {
                if got < want - 1e-9 {
                    worse.push(format!(
                        "{class} {metric}: rust {got:.3} < python {want:.3}"
                    ));
                }
            }
        }
    }
    worse
}

/// The configuration Python's bench runs: the dense arm is pgvector, the
/// lexical arm is SQLite LIKE (no BM25 backend). Same page, 53 of 53.
#[test]
fn pg_vector_arm_matches_python_pages_page_for_page() {
    let Some(dsn) = dsn() else { return };
    let res = run(&dsn, false, true);
    let d = page_diffs(&res);
    assert!(d.is_empty(), "{} pages differ:\n{}", d.len(), d.join("\n"));
    assert!(worse(&res).is_empty(), "{:#?}", worse(&res));
    assert_eq!(res.leak_count, 0);
    eprintln!("pg vector + LIKE: 53/53 pages identical to Python");
}

/// pgvector exactly as it is (no cosine floor) with SQLite LIKE: the head of
/// every page agrees with Python, the tail may carry zero-similarity rows that
/// KNN returns and the in-memory bench arm does not. Gate: no class metric below
/// the golden, no leak; the page agreement is reported.
#[test]
fn pg_vector_arm_without_floor_meets_or_beats_python_per_class() {
    let Some(dsn) = dsn() else { return };
    let res = run(&dsn, false, false);
    let d = page_diffs(&res);
    eprintln!(
        "pg vector (no floor) + LIKE: {}/{} pages identical to Python",
        res.rows.len() - d.len(),
        res.rows.len()
    );
    assert!(worse(&res).is_empty(), "{:#?}", worse(&res));
    assert_eq!(res.leak_count, 0);
}

/// The production arm set: pgvector plus pg_search BM25 as `lexical`. The
/// pinned lexical arm changes (BM25 replaces LIKE), so pages may differ from the
/// LIKE golden: the gate is that no per-class metric falls below it and that
/// nothing leaks.
#[test]
fn full_pg_arm_set_meets_or_beats_python_per_class() {
    let Some(dsn) = dsn() else { return };
    let res = run(&dsn, true, false);
    let d = page_diffs(&res);
    eprintln!(
        "pg vector + BM25: {}/{} pages identical to Python",
        res.rows.len() - d.len(),
        res.rows.len()
    );
    eprintln!("{}", bench::format_table(&res));
    assert!(worse(&res).is_empty(), "{:#?}", worse(&res));
    assert_eq!(res.leak_count, 0);
}

/// A read-only configuration opens every connection with
/// `default_transaction_read_only=on`: a write on it is refused by the server.
#[test]
fn read_only_config_refuses_writes() {
    let Some(dsn) = dsn() else { return };
    with_schema(&dsn, false, |schema, fac| {
        let cfg = fac.config(false, true);
        let mut c = cfg.connect_config().unwrap().connect(NoTls).unwrap();
        let ro: String = c
            .query_one("SHOW default_transaction_read_only", &[])
            .unwrap()
            .get(0);
        assert_eq!(ro, "on");
        let e = c
            .execute(
                &format!("INSERT INTO {schema}.memories (id, memory_json) VALUES ('x', '{{}}')"),
                &[],
            )
            .unwrap_err();
        let msg = e
            .as_db_error()
            .map(|d| d.message().to_string())
            .unwrap_or_default();
        assert!(
            msg.contains("read-only"),
            "expected a read-only refusal, got: {msg}"
        );
        // And the writable config can.
        let mut w = fac
            .config(false, false)
            .connect_config()
            .unwrap()
            .connect(NoTls)
            .unwrap();
        w.execute(
            &format!("INSERT INTO {schema}.memories (id, memory_json) VALUES ('x', '{{}}')"),
            &[],
        )
        .unwrap();
    });
}
