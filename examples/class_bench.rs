//! Print the per-class bench table: `cargo run --example class_bench [fixture-dir]`.
//! The default directory is `tests/fixtures/eval-v1` of this crate.

fn main() {
    let dir = std::env::args().nth(1).map_or_else(
        || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval-v1"),
        std::path::PathBuf::from,
    );
    match skmem::bench::run_bench(&dir) {
        Ok(res) => print!("{}", skmem::bench::format_table(&res)),
        Err(e) => {
            eprintln!("class_bench: {e}");
            std::process::exit(1);
        }
    }
}
