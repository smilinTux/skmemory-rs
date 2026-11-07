//! The few libc calls the intention store needs, declared directly so the
//! crate takes no new dependency: `flock(2)` (the same advisory lock Python's
//! `fcntl.flock` takes, which is what lets both engines serialise on one
//! lock file), `getuid` and `gethostname`.

use std::fs::File;
use std::io;
use std::os::raw::{c_char, c_int};
use std::os::unix::io::AsRawFd;

extern "C" {
    fn flock(fd: c_int, operation: c_int) -> c_int;
    fn getuid() -> u32;
    fn gethostname(name: *mut c_char, len: usize) -> c_int;
}

const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;
const LOCK_UN: c_int = 8;

/// Outcome of a non-blocking lock attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum TryLock {
    Acquired,
    /// Another open file description holds the lock (`EWOULDBLOCK`).
    Busy,
}

fn call(f: &File, op: c_int) -> io::Result<()> {
    loop {
        // SAFETY: the descriptor is owned by `f` and valid for the call.
        let r = unsafe { flock(f.as_raw_fd(), op) };
        if r == 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// `flock(fd, LOCK_EX)`: block until acquired (EINTR is retried).
pub fn lock_exclusive(f: &File) -> io::Result<()> {
    call(f, LOCK_EX)
}

/// `flock(fd, LOCK_EX | LOCK_NB)`.
pub fn try_lock_exclusive(f: &File) -> io::Result<TryLock> {
    match call(f, LOCK_EX | LOCK_NB) {
        Ok(()) => Ok(TryLock::Acquired),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(TryLock::Busy),
        Err(e) => Err(e),
    }
}

/// `flock(fd, LOCK_UN)`.
pub fn unlock(f: &File) -> io::Result<()> {
    call(f, LOCK_UN)
}

/// `os.getuid()`.
pub fn uid() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { getuid() }
}

/// `socket.gethostname()`.
pub fn hostname() -> String {
    let mut buf = [0 as c_char; 256];
    // SAFETY: the buffer is valid for `len` bytes; the result is NUL-terminated on success.
    let r = unsafe { gethostname(buf.as_mut_ptr(), buf.len() - 1) };
    if r != 0 {
        return "localhost".into();
    }
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// SHA-1 as lowercase hex. Used only to name the lock directory the way the
/// Python store does (`hashlib.sha1(str(path)).hexdigest()[:16]`); it is not
/// a security boundary.
pub fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for (i, c) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(&[b'a'; 1000]),
            "291e9a6c66994949b57ba5e650361e98fc36b1ba"
        );
    }

    #[test]
    fn two_descriptions_conflict() {
        let d = std::env::temp_dir().join(format!("skmem-sys-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("l.lock");
        let a = File::create(&p).unwrap();
        let b = File::open(&p).unwrap();
        assert_eq!(try_lock_exclusive(&a).unwrap(), TryLock::Acquired);
        assert_eq!(try_lock_exclusive(&b).unwrap(), TryLock::Busy);
        unlock(&a).unwrap();
        assert_eq!(try_lock_exclusive(&b).unwrap(), TryLock::Acquired);
        let _ = std::fs::remove_dir_all(&d);
    }
}
