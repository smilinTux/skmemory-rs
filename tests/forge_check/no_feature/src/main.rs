//! Must NOT compile: without the `testing` feature the env-injecting scope
//! constructors do not exist, so a caller cannot ignore the process ceiling.
use skmem::audience::{AudienceConfig, OwnerScope, ReaderScope};
use skmem::env::Env;

fn main() {
    let _owner = OwnerScope::resolve(&Env::empty());
    let _scope = ReaderScope::resolve(None, &Env::empty(), &AudienceConfig::empty());
}
