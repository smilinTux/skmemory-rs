//! SKMEM-R-06D review round 1: the REAL process ceiling cannot be bypassed.
//!
//! One test, in its own process (it sets `SKMEMORY_READER`). With the ceiling at
//! `@public`, no public path may yield an `OwnerScope` or an unrestricted
//! `ReaderScope`, whatever per-call reader or audience config is passed. The
//! env-injecting constructors (`resolve`) exist only under the `testing`
//! feature; `tests/forge_check` (run by CI) proves a build without it cannot
//! call them.

use skmem::audience::{AudienceConfig, OwnerScope, ReaderScope, READER_ENV};
use skmem::flat::Store;
use skmem::model::Memory;

#[test]
fn a_process_ceiling_cannot_be_bypassed_from_the_public_api() {
    let dir = std::env::temp_dir().join(format!("skmem-noforge-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // seed as the owner BEFORE the ceiling is declared
    std::env::remove_var(READER_ENV);
    let own = OwnerScope::from_process().expect("no ceiling: owner");
    let store = Store::open(&dir).unwrap();
    let mut secret = Memory::new("secret", "hidden body");
    secret.id = "secret".into();
    secret.context_tag = "@chef-only".into();
    store.save(&secret, &own).unwrap();
    assert!(store.load("secret", own.reader_scope()).unwrap().is_some());

    // a config under which a channel member has full trust and an owner-only tag
    let rich = AudienceConfig::from_json(
        r#"{"channels":{"c":{"context_tag":"@chef-only","members":["A"]}},
            "people":{"A":{"trust_level":4}}}"#,
    );
    for ceiling in [
        "@public",
        "@community",
        "@inner-circle",
        "channel:unknown",
        "garbage",
    ] {
        std::env::set_var(READER_ENV, ceiling);
        assert!(
            OwnerScope::from_process().is_err(),
            "{ceiling}: from_process minted an OwnerScope"
        );
        for reader in [
            None,
            Some("owner"),
            Some("@chef-only"),
            Some("@chef"),
            Some("  OWNER "),
            Some("@inner-circle"),
            Some("channel:c"),
            Some("root"),
            Some(""),
        ] {
            for sc in [
                ReaderScope::from_process(reader),
                ReaderScope::from_process_with_config(reader, &rich),
                ReaderScope::from_process_with_config(reader, &AudienceConfig::empty()),
            ] {
                // never wider than a public reader, and the ceiling `@public`
                // (the strictest here) must hold for every reader
                assert!(!sc.unrestricted(), "{ceiling}/{reader:?} is unrestricted");
                assert!(!sc.allows(None, &[]), "{ceiling}/{reader:?}");
                if ceiling == "@public" || ceiling == "garbage" {
                    assert!(!sc.allows(Some("@chef-only"), &[]), "{ceiling}/{reader:?}");
                    assert!(!sc.allows(Some("@community"), &[]), "{ceiling}/{reader:?}");
                }
                assert!(
                    store.load("secret", &sc).unwrap().is_none(),
                    "{ceiling}/{reader:?} read a hidden memory"
                );
            }
        }
    }
    // a blank ceiling is not a declaration, and the owner spellings are owners
    for ok in ["", "   ", "owner", "@chef-only"] {
        std::env::set_var(READER_ENV, ok);
        assert!(OwnerScope::from_process().is_ok(), "{ok:?}");
    }
    std::env::remove_var(READER_ENV);
    let _ = std::fs::remove_dir_all(&dir);
}
