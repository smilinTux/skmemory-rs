//! The `skmem-mcp` binary: an MCP server over stdio. See [`skmem::mcp`].
fn main() {
    std::process::exit(skmem::mcp::run_main());
}
