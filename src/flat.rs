//! Flat JSON store: one `<id>.json` per memory under
//! `<root>/{short-term,mid-term,long-term}/`, the same layout as Python's
//! `FileBackend`, with durable atomic writes (temp file, fsync, rename,
//! directory fsync) like `skmemory.atomic_io`.

use crate::audience::{OwnerScope, ReaderScope};
use crate::model::{Layer, Memory, ModelError};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Syncthing conflict marker inside a file name.
pub const CONFLICT_MARK: &str = ".sync-conflict-";

/// Errors from the flat store.
#[derive(Debug)]
pub enum StoreError {
    Io(io::Error),
    /// The file exists but is not a valid memory (Python's `load` returns None).
    Invalid(ModelError),
    /// An id that would escape the layer directory or resolve to `.json`.
    BadId(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "io: {e}"),
            StoreError::Invalid(e) => write!(f, "invalid record: {e}"),
            StoreError::BadId(s) => write!(f, "bad memory id: {s:?}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        StoreError::Io(e)
    }
}

/// Where a durable write stops, for crash simulation in tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashPoint {
    /// Temp file written and flushed, not yet fsynced or renamed.
    AfterTempWrite,
    /// Temp file fsynced, not yet renamed over the target.
    AfterFsync,
}

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Durably write `data` to `path`: temp file in the same directory, fsync,
/// atomic rename, best-effort directory fsync. On any error the temp file is
/// removed and the prior content of `path` is untouched.
pub fn atomic_write_bytes(path: &Path, data: &[u8]) -> io::Result<()> {
    atomic_write_inner(path, data, None)
}

/// Like [`atomic_write_bytes`] but simulates a crash at `crash`: the temp file
/// is left behind (as a killed process would) and an error is returned.
pub fn atomic_write_crashing(path: &Path, data: &[u8], crash: CrashPoint) -> io::Result<()> {
    atomic_write_inner(path, data, Some(crash))
}

fn atomic_write_inner(path: &Path, data: &[u8], crash: Option<CrashPoint>) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(".{name}.tmp-{}-{n:08x}", std::process::id()));
    let result = (|| {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(data)?;
        f.flush()?;
        if crash == Some(CrashPoint::AfterTempWrite) {
            return Err(io::Error::other("simulated crash after temp write"));
        }
        f.sync_all()?;
        if crash == Some(CrashPoint::AfterFsync) {
            return Err(io::Error::other("simulated crash after fsync"));
        }
        fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        if crash.is_none() {
            let _ = fs::remove_file(&tmp);
        }
        return Err(e);
    }
    // Best effort: make the rename entry itself durable.
    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Python `require_memory_id`, plus a guard against path separators.
fn check_id(id: &str) -> Result<&str, StoreError> {
    let t = crate::model::py_strip(id);
    if t.is_empty() || t.contains('/') || t.contains('\\') || t == "." || t == ".." {
        return Err(StoreError::BadId(id.to_string()));
    }
    Ok(t)
}

fn conflict_copy_id(name: &str) -> Option<&str> {
    name.contains(CONFLICT_MARK)
        .then(|| name.split(CONFLICT_MARK).next().unwrap_or(name))
}

/// Visible `*.json` entries of a directory (Python `glob("*.json")`: dotfiles
/// such as in-flight temp files are not matched).
pub(crate) fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(Result::ok)
            .filter(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.ends_with(".json") && !n.starts_with('.')
            })
            .map(|e| e.path())
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

/// A file `list` could not turn into a memory.
#[derive(Debug, Clone)]
pub struct Unloadable {
    pub path: PathBuf,
    pub reason: String,
}

/// Result of [`Store::list`]: what loaded, and what did not and why.
#[derive(Debug, Clone, Default)]
pub struct Listing {
    pub memories: Vec<Memory>,
    pub unloadable: Vec<Unloadable>,
}

fn fsync_dir(dir: &Path) {
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

/// A flat memory store rooted at `<agent>/memory/`.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Open (creating the three layer directories, like Python `_ensure_dirs`).
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Store> {
        let root = root.into();
        for l in Layer::ALL {
            fs::create_dir_all(root.join(l.as_str()))?;
        }
        Ok(Store { root })
    }

    /// Open without creating anything (for read-only inspection).
    pub fn open_readonly(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn layer_dir(&self, layer: Layer) -> PathBuf {
        self.root.join(layer.as_str())
    }

    /// Path a memory is saved to.
    pub fn path_for(&self, memory: &Memory) -> Result<PathBuf, StoreError> {
        let id = check_id(&memory.id)?;
        Ok(self.layer_dir(memory.layer).join(format!("{id}.json")))
    }

    /// Save with a durable atomic write. Returns the id.
    pub fn save(&self, memory: &Memory) -> Result<String, StoreError> {
        let path = self.path_for(memory)?;
        atomic_write_bytes(&path, memory.to_json().as_bytes())?;
        Ok(memory.id.clone())
    }

    /// Locate a memory file: the canonical `<id>.json` in any layer, else the
    /// newest Syncthing conflict copy (the record's only copy).
    ///
    /// Crate-private: a path answers "does this id exist" for any reader, so
    /// the public lookups ([`Store::load`], [`Store::flat_id_set`]) take a
    /// [`ReaderScope`] and a hidden id reads exactly like a missing one.
    pub(crate) fn find_file(&self, id: &str) -> Option<PathBuf> {
        let id = check_id(id).ok()?;
        for l in Layer::ALL {
            let p = self.layer_dir(l).join(format!("{id}.json"));
            if p.exists() {
                return Some(p);
            }
        }
        for l in Layer::ALL {
            let prefix = format!("{id}{CONFLICT_MARK}");
            let copies: Vec<PathBuf> = json_files(&self.layer_dir(l))
                .into_iter()
                .filter(|p| file_name(p).starts_with(&prefix))
                .collect();
            if let Some(last) = copies.last() {
                return Some(last.clone());
            }
        }
        None
    }

    /// Load by id as `scope` sees it. `Ok(None)` when no file exists OR the
    /// reader may not see the memory (a hidden id is exactly a missing one).
    /// For the owner, `Err(Invalid)` when the file is not a valid memory
    /// (Python returns None for both). A restricted reader gets `Ok(None)` for
    /// an invalid file too: its audience cannot be known, and an error would
    /// say the id exists.
    pub fn load(&self, id: &str, scope: &ReaderScope) -> Result<Option<Memory>, StoreError> {
        let loaded = match self.find_file(id) {
            None => return Ok(None),
            Some(p) => self.read_path(&p),
        };
        match loaded {
            Ok(mut m) => {
                if !scope.allows_memory(&m) {
                    return Ok(None);
                }
                self.redact_links(&mut m, scope, &mut HashMap::new());
                Ok(Some(m))
            }
            Err(_) if !scope.unrestricted() => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Whether `scope` may see the memory `id`; an unknown, unreadable or
    /// malformed id is NOT visible (a dangling link must read like a hidden one).
    fn visible_to(&self, id: &str, scope: &ReaderScope) -> bool {
        self.find_file(id)
            .and_then(|p| self.read_path(&p).ok())
            .is_some_and(|m| scope.allows_memory(&m))
    }

    /// Redact the links of `m` that name memories `scope` cannot see.
    fn redact_links(&self, m: &mut Memory, scope: &ReaderScope, cache: &mut HashMap<String, bool>) {
        scope.redact_links(m, &mut |id| {
            *cache
                .entry(id.to_string())
                .or_insert_with(|| self.visible_to(id, scope))
        });
    }

    /// Read and parse one file, unscoped. Crate-private: the public lookups
    /// take a [`ReaderScope`]. Never writes.
    pub(crate) fn read_path(&self, path: &Path) -> Result<Memory, StoreError> {
        let raw = fs::read(path)?;
        Memory::from_stored_json(&raw).map_err(StoreError::Invalid)
    }

    /// True for a conflict copy that is not the record's only live copy:
    /// a canonical `<id>.json` exists in any layer, or a later-sorted sibling
    /// conflict copy exists.
    pub fn is_hidden_conflict_copy(&self, path: &Path) -> bool {
        let name = file_name(path);
        let Some(cid) = conflict_copy_id(&name) else {
            return false;
        };
        let prefix = format!("{cid}{CONFLICT_MARK}");
        for l in Layer::ALL {
            let d = self.layer_dir(l);
            if d.join(format!("{cid}.json")).exists() {
                return true;
            }
            if json_files(&d)
                .iter()
                .any(|q| q != path && file_name(q).starts_with(&prefix) && file_name(q) > name)
            {
                return true;
            }
        }
        false
    }

    /// List memories (all layers when `layer` is None), newest first. Conflict
    /// copies that are not the only copy are skipped on purpose; a file that
    /// cannot be loaded is never dropped silently: it is reported in
    /// [`Listing::unloadable`] with its path and the reason.
    ///
    /// `scope` filters: a restricted reader gets only memories it may see and an
    /// EMPTY `unloadable` (a broken file's audience is unknowable, and its
    /// presence would be a count leak).
    pub fn list(&self, layer: Option<Layer>, scope: &ReaderScope) -> Listing {
        let mut l = self.list_unscoped(layer);
        l.memories.retain(|m| scope.allows_memory(m));
        if !scope.unrestricted() {
            l.unloadable.clear();
            let mut cache = HashMap::new();
            for m in &mut l.memories {
                self.redact_links(m, scope, &mut cache);
            }
        }
        l
    }

    /// [`Store::list`] without the reader filter, for the crate's own walks.
    pub(crate) fn list_unscoped(&self, layer: Option<Layer>) -> Listing {
        let layers: Vec<Layer> = layer.map_or_else(|| Layer::ALL.to_vec(), |l| vec![l]);
        let mut memories = Vec::new();
        let mut unloadable = Vec::new();
        for l in layers {
            for p in json_files(&self.layer_dir(l)) {
                if self.is_hidden_conflict_copy(&p) {
                    continue;
                }
                match self.read_path(&p) {
                    Ok(m) => memories.push(m),
                    Err(e) => unloadable.push(Unloadable {
                        path: p,
                        reason: e.to_string(),
                    }),
                }
            }
        }
        memories.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));
        Listing {
            memories,
            unloadable,
        }
    }

    /// Filename stems of every flat JSON file across the active layers, as
    /// `scope` sees them: a restricted reader gets only the ids of memories it
    /// may see (each file is read to decide), never the hidden stems.
    pub fn flat_id_set(&self, scope: &ReaderScope) -> std::collections::BTreeSet<String> {
        if scope.unrestricted() {
            return self.flat_id_set_unscoped();
        }
        let mut s = std::collections::BTreeSet::new();
        for l in Layer::ALL {
            for p in json_files(&self.layer_dir(l)) {
                if let Ok(m) = self.read_path(&p) {
                    if scope.allows_memory(&m) {
                        s.insert(file_name(&p).trim_end_matches(".json").to_string());
                    }
                }
            }
        }
        s
    }

    /// Every stem, for the crate's own reconciliation.
    pub(crate) fn flat_id_set_unscoped(&self) -> std::collections::BTreeSet<String> {
        let mut s = std::collections::BTreeSet::new();
        for l in Layer::ALL {
            for p in json_files(&self.layer_dir(l)) {
                let n = file_name(&p);
                s.insert(n.trim_end_matches(".json").to_string());
            }
        }
        s
    }

    /// Delete the memory file (and nothing else). Returns whether one existed,
    /// which is an existence oracle for a hidden id: owner only.
    pub fn delete(&self, id: &str, _owner: &OwnerScope) -> Result<bool, StoreError> {
        match self.find_file(id) {
            None => Ok(false),
            Some(p) => {
                fs::remove_file(&p)?;
                if let Some(parent) = p.parent() {
                    fsync_dir(parent);
                }
                Ok(true)
            }
        }
    }

    /// Directory holding tombstones: `<root>/tombstones/`.
    pub fn tombstone_dir(&self) -> PathBuf {
        self.root.join("tombstones")
    }

    /// Write the durable marker `<root>/tombstones/<id>.json`, same payload and
    /// bytes as Python `write_tombstone` (`json.dumps`, compact separators).
    /// `forgotten_at` is supplied by the caller (Python uses local time with
    /// `%Y-%m-%dT%H:%M:%S%z`). Owner only.
    pub fn tombstone(
        &self,
        id: &str,
        forgotten_at: &str,
        reason: &str,
        agent: Option<&str>,
        _owner: &OwnerScope,
    ) -> Result<PathBuf, StoreError> {
        let id = check_id(id)?;
        let mut m = serde_json::Map::new();
        m.insert("id".into(), id.into());
        m.insert("forgotten_at".into(), forgotten_at.into());
        m.insert("reason".into(), reason.into());
        if let Some(a) = agent.filter(|a| !a.is_empty()) {
            m.insert("agent".into(), a.into());
        }
        let path = self.tombstone_dir().join(format!("{id}.json"));
        let body = crate::pyjson::dumps_compact(&serde_json::Value::Object(m));
        atomic_write_bytes(&path, body.as_bytes())?;
        Ok(path)
    }

    /// True when a tombstone exists for `id`: a forgotten id reads differently
    /// from a never-seen one, so owner only.
    pub fn is_tombstoned(&self, id: &str, _owner: &OwnerScope) -> bool {
        check_id(id)
            .map(|id| self.tombstone_dir().join(format!("{id}.json")).exists())
            .unwrap_or(false)
    }

    /// Move a memory file out of the active tiers into cold storage:
    /// `<root>/archive/<id>.json`, or `<root>/archive/<sub>/<id>.json`
    /// (Python uses `deduped` as the sub directory). The bytes are not
    /// rewritten. Returns the new path, or None if the memory was not found.
    /// Returns a path (None when not found), so it is an existence oracle and a
    /// write: owner only.
    pub fn archive(
        &self,
        id: &str,
        sub: Option<&str>,
        _owner: &OwnerScope,
    ) -> Result<Option<PathBuf>, StoreError> {
        let Some(src) = self.find_file(id) else {
            return Ok(None);
        };
        let mut dir = self.root.join("archive");
        if let Some(s) = sub {
            dir = dir.join(check_id(s)?);
        }
        fs::create_dir_all(&dir)?;
        let dest = dir.join(file_name(&src));
        fs::rename(&src, &dest)?;
        // Make both directory entries durable: the new one and the removed one.
        fsync_dir(&dir);
        if let Some(parent) = src.parent() {
            fsync_dir(parent);
        }
        Ok(Some(dest))
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}
