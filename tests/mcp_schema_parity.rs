//! SKMEM-R-09: the Rust MCP server's tool schemas equal the Python server's.
//!
//! `tests/fixtures/mcp/python-tools.json` is generated against the Python
//! package by `scripts/gen_mcp_tool_fixture.py` (committed). For every tool the
//! Rust server serves, name, description and input schema must be equal; every
//! other Python tool must be on the explicit not-served list; and the reader
//! policy (owner-only or not, which tools take a `reader`) must agree.

use serde_json::Value;
use skmem::mcp::{all_tool_names, is_owner_only, reader_tools, tool_list, NOT_SERVED, SERVED};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp/python-tools.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn py_tools() -> BTreeMap<String, Value> {
    fixture()["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["name"].as_str().unwrap().to_string(), t.clone()))
        .collect()
}

#[test]
fn served_tools_equal_python_name_description_and_schema() {
    let py = py_tools();
    let rust = tool_list();
    assert_eq!(rust.len(), SERVED.len());
    for t in &rust {
        let name = t["name"].as_str().unwrap();
        let want = py
            .get(name)
            .unwrap_or_else(|| panic!("{name}: not a Python tool"));
        assert_eq!(t, want, "{name}: schema differs from the Python server");
    }
    let listed: Vec<&str> = rust.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(listed, SERVED.to_vec(), "tools/list order");
}

#[test]
fn served_plus_not_served_is_exactly_the_python_tool_set() {
    let py: BTreeSet<String> = py_tools().into_keys().collect();
    let ours: BTreeSet<String> = all_tool_names().into_iter().map(String::from).collect();
    assert_eq!(
        ours, py,
        "a Python tool is neither served nor listed as left to Python"
    );
    assert_eq!(all_tool_names().len(), py.len(), "a tool is listed twice");
}

#[test]
fn reader_policy_matches_python() {
    let f = fixture();
    let policy = f["reader_policy"].as_object().unwrap();
    for name in all_tool_names() {
        let class = policy[name]["class"].as_str().unwrap();
        assert_eq!(
            is_owner_only(name),
            class == "owner-only",
            "{name}: owner-only classification"
        );
    }
    let py_reader: Vec<&str> = f["reader_tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut ours = reader_tools().to_vec();
    ours.sort_unstable();
    assert_eq!(ours, py_reader, "tools that gain a `reader` property");
    // The served tools that take a reader carry the property in the Python schema.
    for t in tool_list() {
        let name = t["name"].as_str().unwrap();
        if reader_tools().contains(&name) {
            assert!(
                t["inputSchema"]["properties"]["reader"].is_object(),
                "{name}"
            );
        }
    }
}

#[test]
fn not_served_list_has_no_served_tool() {
    for (n, _) in NOT_SERVED {
        assert!(!SERVED.contains(&n), "{n}");
    }
}
