use std::process::Command;

#[test]
fn bin_prints_crate_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_skmem"))
        .output()
        .expect("run skmem");
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        env!("CARGO_PKG_VERSION")
    );
}
