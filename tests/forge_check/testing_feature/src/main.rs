//! Must NOT build: a consumer enabling `testing` would get the constructors
//! that bypass the process reader ceiling, so skmem refuses to compile.
use skmem::audience::OwnerScope;
use skmem::env::Env;

fn main() {
    let _owner = OwnerScope::resolve(&Env::empty());
}
