//! SKMEM-R-09: a process-wide `SKMEMORY_READER` is a ceiling the MCP server keeps.
//!
//! This file mutates the process environment, so it holds ONE test function.

use serde_json::{json, Value};
use skmem::engine::{Arms, Engine, EngineOptions};
use skmem::mcp::Server;

fn text(s: &Server, name: &str, args: Value) -> String {
    s.call_tool(name, &args)["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn declared_reader_ceiling_refuses_owner_only_and_cannot_be_widened() {
    let dir = std::env::temp_dir().join(format!("skmem-mcp-ceil-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("SKMEMORY_READER", "@public");
    let engine = Engine::open(dir.join("memory"), EngineOptions::new(Arms::none())).unwrap();
    assert!(engine.is_read_only(), "a reader process opens read-only");
    let s = Server::with_engine(engine);

    for tool in ["memory_remind", "memory_intentions", "memory_intention_done", "memory_promote"] {
        assert_eq!(
            text(&s, tool, json!({"title": "x", "due_at": "in 1 day", "id": "a"})),
            format!(r#"{{"error": "{tool} is owner-only and cannot be used by a declared reader"}}"#)
        );
        // naming the owner per call does not widen the ceiling
        assert!(text(&s, tool, json!({"reader": "owner", "title": "x", "due_at": "in 1 day"}))
            .contains("owner-only and cannot be used by a declared reader"));
    }

    // Reads still work, narrowed; a store is refused and nothing is written.
    assert_eq!(text(&s, "memory_list", json!({"reader": "owner"})), "[]");
    let stored = text(&s, "memory_store", json!({"title": "t", "content": "c"}));
    assert!(stored.contains("error"), "{stored}");
    assert!(!dir.join("memory").join("short-term").exists());

    std::env::remove_var("SKMEMORY_READER");
    let _ = std::fs::remove_dir_all(&dir);
}
