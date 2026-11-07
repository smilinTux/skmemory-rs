//! Must NOT compile: without the `testing` feature `ReaderScope::resolve` does
//! not exist, so a caller cannot ignore the process ceiling.
use skmem::audience::{AudienceConfig, ReaderScope};
use skmem::env::Env;

fn main() {
    let _scope = ReaderScope::resolve(None, &Env::empty(), &AudienceConfig::empty());
}
