//! Parity with Python skmemory's fusion and post-fusion stages.
//!
//! `tests/golden/fusion_parity.json` is written by `tests/golden/gen_fusion_golden.py`
//! from the CURRENT Python (`skmemory.fusion`, `rerank`, `tiers`, `procedures`,
//! `decay`, and `MemoryStore.search` over fake backends). Every case here must
//! reproduce Python exactly: same ids, same order, same RRF scores. The data is
//! synthetic. Hermetic: no Python, no network, no environment reads (each case
//! carries its own settings in an `Env`).

use serde_json::Value;
use skmem::decay::{self, Channel, Usage};
use skmem::env::Env;
use skmem::fusion::{self, ArmInput, ArmOutcome, Route, RouteDecision, RRF_K};
use skmem::pytime::Dt;
use skmem::rerank::Reranker;
use skmem::search::{demote_superseded, search, Candidate, SearchInput};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn golden() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/fusion_parity.json");
    serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap()
}

fn section(g: &Value, name: &str) -> Vec<Value> {
    g[name]
        .as_array()
        .unwrap_or_else(|| panic!("{name}"))
        .clone()
}

fn s(v: &Value) -> String {
    v.as_str().unwrap().to_string()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(s).collect()
}

fn env_of(v: &Value) -> Env {
    let mut e = Env::empty();
    if let Some(o) = v.as_object() {
        for (k, val) in o {
            e = e.with(k, val.as_str().unwrap());
        }
    }
    e
}

fn arm_input(spec: &Value) -> ArmInput {
    match spec {
        Value::Null => ArmInput::Unattached,
        Value::String(x) if x == "disabled" => ArmInput::Disabled("off by config".into()),
        Value::Object(o) => ArmInput::Failed(s(&o["raise"])),
        Value::Array(_) => ArmInput::Ids(strs(spec)),
        other => panic!("arm spec {other}"),
    }
}

fn check_outcomes(name: &str, got: &[ArmOutcome], want: &Value) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{name}: arm count");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.name, s(&w["name"]), "{name}");
        assert_eq!(g.status.as_str(), s(&w["status"]), "{name}: {}", g.name);
        assert_eq!(g.ids, strs(&w["ids"]), "{name}: {} ids", g.name);
    }
}

fn route_of(v: &Value) -> Option<Route> {
    match v.as_str() {
        Some("identifier") => Some(Route::Identifier),
        Some("natural") => Some(Route::Natural),
        _ => None,
    }
}

fn candidate(d: &Value) -> Candidate {
    let opt = |k: &str| d.get(k).and_then(|v| v.as_str()).map(str::to_string);
    let id = s(&d["id"]);
    Candidate {
        title: opt("title").filter(|t| !t.is_empty()).unwrap_or(id.clone()),
        id,
        content: opt("content").unwrap_or_default(),
        created_at: Some(opt("created_at").unwrap_or("2026-10-01T00:00:00+00:00".into())),
        layer: opt("layer").unwrap_or("short-term".into()),
        intensity: d.get("intensity").and_then(Value::as_f64).unwrap_or(0.0),
        kind: opt("kind").unwrap_or("fact".into()),
        outcome: opt("outcome"),
        valid_to: opt("valid_to"),
        supersedes_id: opt("supersedes_id"),
        superseded_by_id: opt("superseded_by_id"),
        source: opt("source").unwrap_or("manual".into()),
        source_tier: Some(opt("source_tier").unwrap_or("user-stated".into())),
        schema_version: d.get("schema_version").and_then(Value::as_i64).unwrap_or(2),
        context_tag: "@chef-only".into(),
        tags: vec![],
    }
}

fn found_of(docs: &Value) -> HashMap<String, Candidate> {
    docs.as_array()
        .unwrap()
        .iter()
        .map(|d| (s(&d["id"]), candidate(d)))
        .collect()
}

fn prov_of(v: &Value) -> std::collections::BTreeMap<String, usize> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, r)| (k.clone(), r.as_u64().unwrap() as usize))
        .collect()
}

#[test]
fn route_query_matches_python() {
    let g = golden();
    let cases = section(&g, "route");
    assert!(cases.len() >= 40);
    for c in &cases {
        let df: Option<HashMap<String, i64>> = c["df"].as_object().map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
                .collect()
        });
        let f = |t: &str| df.as_ref().and_then(|m| m.get(t)).copied().unwrap_or(0);
        let fref: Option<&dyn Fn(&str) -> i64> = df.as_ref().map(|_| &f as &dyn Fn(&str) -> i64);
        let got = fusion::route_query(&s(&c["query"]), fref, None, &env_of(&c["env"]));
        assert_eq!(got.cls.as_str(), s(&c["cls"]), "{}", c["query"]);
        assert_eq!(got.reason, s(&c["reason"]), "{}", c["query"]);
    }
}

#[test]
fn fuse_matches_python() {
    let g = golden();
    let cases = section(&g, "fuse");
    assert!(cases.len() >= 150);
    for c in &cases {
        let name = s(&c["name"]);
        let arms: Vec<ArmOutcome> = c["arms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                fusion::run_arm(
                    &s(&a["name"]),
                    arm_input(&a["spec"]),
                    fusion::DEFAULT_ARM_CAP,
                )
            })
            .collect();
        check_outcomes(&name, &arms, &c["arm_outcomes"]);
        let pins = strs(&c["pin_arms"]);
        let pin_refs: Vec<&str> = pins.iter().map(String::as_str).collect();
        let route = route_of(&c["route"]).map(|cls| RouteDecision {
            cls,
            reason: "golden",
        });
        let k = c["k"].as_i64().unwrap_or(RRF_K);
        let got = fusion::fuse(
            &arms,
            c["limit"].as_u64().unwrap() as usize,
            k,
            &s(&c["strategy"]),
            &pin_refs,
            route.as_ref(),
            &env_of(&c["env"]),
        );
        assert_eq!(got.ids, strs(&c["ids"]), "{name}: ids");
        let want_prov: std::collections::BTreeMap<_, _> = c["provenance"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), prov_of(v)))
            .collect();
        assert_eq!(got.provenance, want_prov, "{name}: provenance");
        assert_eq!(got.route, route_of(&c["result_route"]), "{name}: route");
    }
}

fn check_scored(name: &str, got: &[skmem::fusion::Scored], want: &Value) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{name}: len");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.id, s(&w[0]), "{name}: id");
        assert_eq!(g.score, w[1].as_f64().unwrap(), "{name}: score of {}", g.id);
        assert_eq!(g.prov, prov_of(&w[2]), "{name}: prov of {}", g.id);
    }
}

#[test]
fn rrf_interleave_and_pinned_scores_match_python() {
    let g = golden();
    let cases = section(&g, "scores");
    assert!(cases.len() >= 20);
    for (n, c) in cases.iter().enumerate() {
        let arms: fusion::ArmResults = c["arms"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), strs(v)))
            .collect();
        let k = c["k"].as_i64().unwrap();
        let env = env_of(&c["env"]);
        let name = format!("scores-{n}");
        check_scored(&name, &fusion::reciprocal_rank_fusion(&arms, k), &c["rrf"]);
        check_scored(&name, &fusion::rank_interleave(&arms, k), &c["interleave"]);
        let pa = strs(&c["pin_arms"]);
        let sa = strs(&c["slot_arms"]);
        let pa: Vec<&str> = pa.iter().map(String::as_str).collect();
        let sa: Vec<&str> = sa.iter().map(String::as_str).collect();
        let pin = c["pin"].as_u64().map(|p| p as usize);
        check_scored(
            &name,
            &fusion::pinned_rrf(&arms, k, pin, &pa, &sa, &env),
            &c["pinned"],
        );
    }
}

#[test]
fn rare_term_reranker_matches_python() {
    let g = golden();
    let cases = section(&g, "rerank");
    assert!(cases.len() >= 40);
    for c in &cases {
        let name = s(&c["name"]);
        let r = Reranker::from_env(&env_of(&c["env"]));
        let out = r.rerank(&s(&c["query"]), &strs(&c["docs"]));
        let order: Vec<usize> = c["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(out.order, order, "{name}: order");
        assert_eq!(out.status.as_str(), s(&c["status"]), "{name}: status");
        assert_eq!(out.boost_terms, strs(&c["boost_terms"]), "{name}: terms");
        let boosted: Vec<usize> = c["boosted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(out.boosted, boosted, "{name}: boosted");
    }
}

#[test]
fn supersession_demotion_matches_python() {
    let g = golden();
    let cases = section(&g, "demote");
    assert!(cases.len() >= 40);
    for c in &cases {
        let found = found_of(&c["docs"]);
        let got = demote_superseded(&strs(&c["ids"]), &found);
        assert_eq!(got, strs(&c["out"]), "{}", c["name"]);
    }
}

#[test]
fn tier_cap_matches_python() {
    let g = golden();
    let cases = section(&g, "tiercap");
    assert!(cases.len() >= 40);
    for c in &cases {
        let found = found_of(&c["docs"]);
        let got = skmem::tiers::apply_tier_cap(
            &strs(&c["ids"]),
            &found,
            c["limit"].as_u64().unwrap() as usize,
            &env_of(&c["env"]),
        );
        assert_eq!(got, strs(&c["out"]), "{}", c["name"]);
    }
}

#[test]
fn procedure_order_matches_python() {
    let g = golden();
    let cases = section(&g, "procedures");
    assert!(cases.len() >= 25);
    for c in &cases {
        let docs = &c["items"];
        let found = found_of(docs);
        let ids: Vec<String> = docs
            .as_array()
            .unwrap()
            .iter()
            .map(|d| s(&d["id"]))
            .collect();
        let got = skmem::procedures::order_procedures(&ids, &found, &s(&c["query"]));
        assert_eq!(got, strs(&c["out"]), "{}", c["query"]);
    }
}

fn usage_of(v: &Value) -> Option<HashMap<String, Usage>> {
    v.as_object().map(|o| {
        o.iter()
            .map(|(k, u)| {
                (
                    k.clone(),
                    Usage {
                        access_count: u["access_count"].as_i64().unwrap(),
                        last_accessed: u["last_accessed"].as_str().map(str::to_string),
                    },
                )
            })
            .collect()
    })
}

#[test]
fn decay_channels_match_python() {
    let g = golden();
    let cases = section(&g, "decay");
    assert!(cases.len() >= 25);
    for (n, c) in cases.iter().enumerate() {
        let found = found_of(&c["docs"]);
        let ids: Vec<String> = c["docs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| s(&d["id"]))
            .collect();
        let usage = usage_of(&c["usage"]);
        let ch = Channel::parse(&s(&c["channel"])).unwrap();
        let as_of = Dt::parse(&s(&c["as_of"])).unwrap();
        let got = decay::rerank(&ids, &found, usage.as_ref(), ch, as_of, decay::RERANK_GAMMA);
        assert_eq!(got, strs(&c["out"]), "decay-{n}");
    }
    for c in section(&g, "decay_config") {
        let env = match c["env"].as_str() {
            None => Env::empty(),
            Some(v) => Env::empty().with("SKMEMORY_DECAY", v),
        };
        let got = decay::configured_channel(&env);
        let want = c["channel"].as_str().and_then(Channel::parse);
        assert_eq!(got, want, "SKMEMORY_DECAY={}", c["env"]);
    }
}

#[test]
fn whole_search_path_matches_python_store_search() {
    let g = golden();
    let cases = section(&g, "pipeline");
    assert!(cases.len() >= 60, "{} pipeline cases", cases.len());
    for c in &cases {
        let name = s(&c["name"]);
        let found = found_of(&c["docs"]);
        let a = &c["arms"];
        let inputs = [
            ("semantic", arm_input(&a["semantic"])),
            ("lexical", arm_input(&a["lexical"]["ids"])),
            ("lexical_local", arm_input(&a["lexical_local"])),
            ("graph", arm_input(&a["graph"])),
            ("temporal", arm_input(&a["temporal"])),
        ];
        let arms: Vec<ArmOutcome> = inputs
            .into_iter()
            .map(|(n, i)| fusion::run_arm(n, i, fusion::DEFAULT_ARM_CAP))
            .collect();
        check_outcomes(&name, &arms, &c["expected"]["arm_outcomes"]);
        let df: HashMap<String, i64> = c["df"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
            .collect();
        let dff = |t: &str| df.get(t).copied().unwrap_or(0);
        let usage = usage_of(&c["usage"]);
        let input = SearchInput {
            query: &s(&c["query"]),
            scope: &owner(),
            limit: c["limit"].as_u64().unwrap() as usize,
            arms: &arms,
            found: &found,
            has_pg_bm25: a["lexical"]["src"] == "pg",
            df: Some(&dff),
            usage: usage.as_ref(),
            decay_as_of: Dt::parse(&s(&c["decay_as_of"])).unwrap(),
        };
        let trace = search(&input, &env_of(&c["env"])).unwrap_or_else(|| panic!("{name}: None"));
        let e = &c["expected"];
        assert_eq!(trace.route, route_of(&e["route"]), "{name}: route");
        assert_eq!(
            trace.demoted,
            strs(&e["fused_stage"]),
            "{name}: fused stage"
        );
        assert_eq!(trace.page, strs(&e["page"]), "{name}: page");
    }
}

#[test]
fn golden_case_count() {
    let g = golden();
    let n: usize = [
        "route",
        "fuse",
        "scores",
        "rerank",
        "demote",
        "tiercap",
        "procedures",
        "decay",
        "decay_config",
        "pipeline",
    ]
    .iter()
    .map(|k| g[*k].as_array().unwrap().len())
    .sum();
    assert!(n >= 500, "only {n} golden cases");
}

/// The owner scope: no reader declared and no `SKMEMORY_READER` ceiling.
fn own() -> skmem::audience::OwnerScope {
    skmem::audience::OwnerScope::resolve(&skmem::env::Env::empty()).unwrap()
}

fn owner() -> skmem::audience::ReaderScope {
    own().reader_scope().clone()
}
