//! Must NOT build: the shipped wheel (`extension-module`) must never carry the
//! `testing` constructors that bypass the process reader ceiling.
fn main() {
    println!("{}", skmem::version());
}
