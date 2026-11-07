//! Must NOT compile: without the `testing` feature `OwnerScope::resolve` does
//! not exist, so a caller cannot ignore the process ceiling.
use skmem::audience::OwnerScope;
use skmem::env::Env;

fn main() {
    let _owner = OwnerScope::resolve(&Env::empty());
}
