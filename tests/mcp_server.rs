//! SKMEM-R-09: the MCP server end to end over a temp store (never a real one).
//!
//! The process must NOT declare `SKMEMORY_READER` (that has its own test file,
//! `mcp_reader_ceiling.rs`); per-call readers are exercised here.

use serde_json::{json, Value};
use skmem::audience::OwnerScope;
use skmem::engine::{Arms, Engine, EngineOptions};
use skmem::mcp::Server;
use skmem::model::Memory;
use std::io::Cursor;

struct Fx {
    _dir: tempfile_dir::Dir,
    server: Server,
    engine_root: std::path::PathBuf,
}

/// A tiny temp-dir helper (no new dependency).
mod tempfile_dir {
    pub struct Dir(pub std::path::PathBuf);
    impl Dir {
        pub fn new(tag: &str) -> Dir {
            let p = std::env::temp_dir().join(format!(
                "skmem-mcp-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn fx(tag: &str) -> Fx {
    let dir = tempfile_dir::Dir::new(tag);
    let root = dir.0.join("memory");
    let engine = Engine::open(&root, EngineOptions::new(Arms::none())).unwrap();
    Fx {
        engine_root: root,
        server: Server::with_engine(engine),
        _dir: dir,
    }
}

fn call(s: &Server, name: &str, args: Value) -> Value {
    let r = s.call_tool(name, &args);
    let text = r["content"][0]["text"].as_str().unwrap().to_string();
    assert_eq!(r["content"][0]["type"], "text");
    serde_json::from_str(&text).unwrap()
}

fn raw(s: &Server, name: &str, args: Value) -> String {
    s.call_tool(name, &args)["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

fn save(f: &Fx, id: &str, title: &str, content: &str, ctx: &str, tags: &[&str]) {
    let own = OwnerScope::from_process().unwrap();
    let engine = Engine::open(&f.engine_root, EngineOptions::new(Arms::none())).unwrap();
    let mut m = Memory::new(title, content);
    m.id = id.to_string();
    m.context_tag = ctx.to_string();
    m.tags = tags.iter().map(|t| t.to_string()).collect();
    m.seal();
    engine.save(&m, &own).unwrap();
}

#[test]
fn initialize_and_tools_list_over_stdio() {
    let f = fx("rpc");
    let input = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        json!({"jsonrpc":"2.0","id":3,"method":"ping"}),
        json!({"jsonrpc":"2.0","id":4,"method":"nope"}),
    ]
    .iter()
    .map(|v| v.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    let mut out = Vec::new();
    f.server
        .serve(Cursor::new(format!("{input}\nnot json\n")), &mut out)
        .unwrap();
    let lines: Vec<Value> = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    // initialized is a notification: no reply. 5 requests/junk lines answered.
    assert_eq!(lines.len(), 5);
    assert_eq!(lines[0]["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(lines[0]["result"]["serverInfo"]["name"], "skmemory");
    assert_eq!(lines[1]["result"]["tools"].as_array().unwrap().len(), 8);
    assert_eq!(lines[2]["result"], json!({}));
    assert_eq!(lines[3]["error"]["code"], -32601);
    assert_eq!(lines[4]["error"]["code"], -32700);
}

#[test]
fn store_recall_list_search_round_trip() {
    let f = fx("rt");
    let r = call(
        &f.server,
        "memory_store",
        json!({"title": "Quantum lunch", "content": "we ate zebracake at noon", "tags": ["food"]}),
    );
    assert_eq!(r["stored"], true);
    let id = r["memory_id"].as_str().unwrap().to_string();

    let m = call(&f.server, "memory_recall", json!({"memory_id": id}));
    assert_eq!(m["title"], "Quantum lunch");
    assert_eq!(m["source"], "mcp");
    assert_eq!(m["source_tier"], "user-stated");
    assert_eq!(m["layer"], "short-term");

    let l = call(&f.server, "memory_list", json!({"tags": ["food"]}));
    assert_eq!(l.as_array().unwrap().len(), 1);
    let none = call(&f.server, "memory_list", json!({"tags": ["nope"]}));
    assert_eq!(none, json!([]));

    let s = call(&f.server, "memory_search", json!({"query": "zebracake"}));
    assert_eq!(s[0]["id"], id.as_str());
    assert_eq!(s[0]["superseded"], false);
    let idx = call(
        &f.server,
        "memory_search",
        json!({"query": "zebracake", "detail": "index"}),
    );
    assert_eq!(idx[0]["score_kind"], "rank");
    assert!(idx[0]["est_tokens"].as_u64().unwrap() > 0);

    // A repeat store of the same content is deduplicated to the same id.
    let again = call(
        &f.server,
        "memory_store",
        json!({"title": "other", "content": "we ate zebracake at noon"}),
    );
    assert_eq!(again["memory_id"], id.as_str());
}

#[test]
fn result_text_is_python_json_dumps_indent_2() {
    let f = fx("fmt");
    let t = raw(&f.server, "memory_list", json!({}));
    assert_eq!(t, "[]");
    let e = raw(&f.server, "memory_recall", json!({"memory_id": "zz"}));
    assert_eq!(e, r#"{"error": "Memory not found: zz"}"#);
    let k = raw(&f.server, "memory_recall", json!({}));
    assert_eq!(k, r#"{"error": "'memory_id'"}"#);
    let s = raw(
        &f.server,
        "memory_store",
        json!({"title": "t", "content": "caf\u{e9} zebra"}),
    );
    assert!(s.starts_with("{\n  \"memory_id\": \""), "{s}");
}

#[test]
fn per_call_reader_narrows_and_hidden_reads_as_not_found() {
    let f = fx("rd");
    save(&f, "pub1", "open", "a public zebra fact", "@public", &[]);
    save(
        &f,
        "priv1",
        "closed",
        "a private zebra secret",
        "@chef-only",
        &[],
    );

    // owner sees both, a public reader only the public one.
    let all = call(&f.server, "memory_list", json!({}));
    assert_eq!(all.as_array().unwrap().len(), 2);
    let pubs = call(&f.server, "memory_list", json!({"reader": "@public"}));
    assert_eq!(pubs.as_array().unwrap().len(), 1);
    assert_eq!(pubs[0]["id"], "pub1");

    let hidden = raw(
        &f.server,
        "memory_recall",
        json!({"memory_id": "priv1", "reader": "@public"}),
    );
    let missing = raw(
        &f.server,
        "memory_recall",
        json!({"memory_id": "nonesuch", "reader": "@public"}),
    );
    assert_eq!(
        hidden.replace("priv1", "X"),
        missing.replace("nonesuch", "X")
    );
    assert!(hidden.contains("Memory not found: priv1"));

    let h = raw(
        &f.server,
        "memory_history",
        json!({"memory_id": "priv1", "reader": "@public"}),
    );
    assert!(h.contains("Memory not found"), "{h}");

    let s = call(
        &f.server,
        "memory_search",
        json!({"query": "zebra", "reader": "@public"}),
    );
    let ids: Vec<&str> = s
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["pub1"]);
}

#[test]
fn owner_only_tools_are_refused_to_a_per_call_reader_identically() {
    let f = fx("own");
    for tool in [
        "memory_remind",
        "memory_intentions",
        "memory_intention_done",
        "memory_forget",
        "memory_export",
        "memory_verify",
    ] {
        let t = raw(
            &f.server,
            tool,
            json!({"reader": "@public", "title": "x", "due_at": "in 1 day", "id": "a"}),
        );
        assert_eq!(
            t,
            format!(
                r#"{{"error": "{tool} is owner-only and cannot be used by a declared reader"}}"#
            )
        );
    }
}

#[test]
fn intentions_remind_list_done() {
    let f = fx("int");
    let r = call(
        &f.server,
        "memory_remind",
        json!({"title": "  check   the thing ", "due_at": "in 2 days", "interval_s": 3600}),
    );
    assert_eq!(r["state"], "pending");
    let id = r["id"].as_str().unwrap().to_string();
    assert_eq!(id.len(), 16);

    let list = call(&f.server, "memory_intentions", json!({}));
    assert_eq!(list.as_array().unwrap().len(), 1);
    assert_eq!(list[0]["title"], "check the thing");
    assert_eq!(list[0]["interval_s"], 3600);

    let done = call(&f.server, "memory_intention_done", json!({"id": id}));
    assert_eq!(done, json!({"id": id, "state": "done"}));
    let pending = call(&f.server, "memory_intentions", json!({"state": "pending"}));
    assert_eq!(pending, json!([]));

    assert_eq!(
        call(&f.server, "memory_intention_done", json!({"id": "nope"})),
        json!({"error": "Intention not found: nope"})
    );
    assert_eq!(
        call(
            &f.server,
            "memory_remind",
            json!({"title": "", "due_at": "in 1 day"})
        ),
        json!({"error": "title is required"})
    );
    assert_eq!(
        call(
            &f.server,
            "memory_remind",
            json!({"title": "x", "due_at": "in 1 day", "interval_s": 5})
        ),
        json!({"error": "interval_s must be at least 60 seconds"})
    );
    let bad = call(
        &f.server,
        "memory_remind",
        json!({"title": "x", "due_at": "someday"}),
    );
    assert!(bad["error"].is_string());
}

#[test]
fn history_walks_the_supersession_chain() {
    let f = fx("hist");
    save(&f, "old1", "v1", "first version", "", &[]);
    let own = OwnerScope::from_process().unwrap();
    let engine = Engine::open(&f.engine_root, EngineOptions::new(Arms::none())).unwrap();
    let mut n = Memory::new("v2", "second version");
    n.id = "new1".into();
    n.supersedes_id = Some("old1".into());
    n.seal();
    engine.save(&n, &own).unwrap();

    let h = call(&f.server, "memory_history", json!({"memory_id": "new1"}));
    let ids: Vec<&str> = h
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["old1", "new1"]);
    assert_eq!(h[1]["requested"], true);
    assert_eq!(h[0]["superseded_by"], "new1");
}

#[test]
fn unserved_and_unknown_tools_say_so() {
    let f = fx("ns");
    let t = call(&f.server, "memory_health", json!({}));
    assert!(t["error"]
        .as_str()
        .unwrap()
        .contains("not served by skmem-mcp"));
    assert_eq!(
        call(&f.server, "memory_nope", json!({})),
        json!({"error": "Unknown tool: memory_nope"})
    );
}
