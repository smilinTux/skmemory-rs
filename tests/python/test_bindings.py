"""Tests for the scoped skmem_core.Engine bindings (SKMEM-R-07B).

Run against the BUILT WHEEL (CI python-wheel job). Scope is read from the
process environment on every call, so most tests flip SKMEMORY_READER with
monkeypatch; the ceiling tests also run in a fresh subprocess so the variable
is set before the extension is even imported.
"""

import json
import os
import subprocess
import sys

import pytest

import skmem_core
from skmem_core import Engine

# Python skmemory `Memory.model_dump()` keys, in order (models.py, schema v2).
MEMORY_KEYS = [
    "id", "created_at", "updated_at", "layer", "role", "title", "content", "summary",
    "tags", "source", "source_ref", "emotional", "related_ids", "parent_id",
    "context_tag", "intent", "integrity_hash", "metadata", "kind", "outcome",
    "event_at", "valid_from", "valid_to", "supersedes_id", "source_tier", "schema_version",
]  # fmt: skip
EMOTIONAL_KEYS = ["intensity", "valence", "labels", "resonance_note", "cloud9_achieved"]
OWNER_ONLY = [
    ("snapshot", ("t", "c")),
    ("save", ({"title": "t", "content": "c"},)),
    ("delete", ("x",)),
    ("archive", ("x",)),
    ("reindex", ()),
    ("remind", ("t", "in 2 days")),
    ("intentions", ()),
    ("intention_done", ("x",)),
    ("ledger_record", ("s", "1", "served")),
    ("ledger_top", ()),
]


@pytest.fixture(autouse=True)
def _no_ceiling(monkeypatch):
    monkeypatch.delenv("SKMEMORY_READER", raising=False)
    monkeypatch.delenv("SKMEMORY_AUDIENCE_CONFIG", raising=False)


@pytest.fixture()
def home(tmp_path):
    return tmp_path / "memory"


@pytest.fixture()
def eng(home):
    return Engine(str(home))


def seed(eng):
    ids = {}
    for name, tag in [("pub", "@public"), ("work", "@work-circle"), ("chef", "@chef-only")]:
        m = eng.snapshot(
            f"{name} note",
            "the amberfox jumps",
            context_tag=tag,
            tags=["fox"],
            layer="long-term" if name == "work" else "short-term",
        )
        ids[name] = m["id"]
    return ids


def test_surface_exposes_no_scope_type():
    names = set(dir(skmem_core)) | set(dir(skmem_core._skmem_core)) | set(dir(Engine))
    for bad in ("OwnerScope", "ReaderScope", "resolve", "from_process", "scope", "Env", "testing"):
        assert bad not in names, bad
    assert skmem_core.version()


def test_memory_dict_matches_python_field_names(eng):
    ids = seed(eng)
    m = eng.recall(ids["pub"])
    assert list(m) == MEMORY_KEYS
    assert list(m["emotional"]) == EMOTIONAL_KEYS
    assert m["layer"] == "short-term" and m["schema_version"] == 2
    assert m["integrity_hash"] and "integrity_warning" not in m["metadata"]
    hit = eng.search("amberfox")[0]
    assert list(hit) == MEMORY_KEYS + ["superseded", "superseded_by"]
    assert hit["superseded"] is False and hit["superseded_by"] is None


def test_python_skmemory_keys_agree_when_installed():
    models = pytest.importorskip("skmemory.models")
    dumped = models.Memory(title="t", content="c").model_dump()
    assert list(dumped) == MEMORY_KEYS


def test_owner_sees_everything_and_filters_work(eng):
    ids = seed(eng)
    assert {h["id"] for h in eng.search("amberfox")} == set(ids.values())
    assert [h["id"] for h in eng.search("amberfox", layer="long-term")] == [ids["work"]]
    assert len(eng.search("amberfox", tags=["fox"], limit=2)) == 2
    assert eng.search("amberfox", tags=["nope"]) == []
    assert eng.search("amberfox", context_tag="@public")[0]["id"] == ids["pub"]
    assert len(eng.list()) == 3
    assert [m["id"] for m in eng.list(layer="long-term")] == [ids["work"]]
    with pytest.raises(ValueError):
        eng.search("amberfox", as_of="not a date")
    with pytest.raises(ValueError):
        eng.search("amberfox", layer="nope")


def test_per_call_reader_narrows(eng):
    ids = seed(eng)
    assert {h["id"] for h in eng.search("amberfox", reader="@public")} == {ids["pub"]}
    assert {h["id"] for h in eng.search("amberfox", reader="@work-circle")} == {
        ids["pub"],
        ids["work"],
    }
    # an unparseable reader is @public, never wider
    assert {h["id"] for h in eng.search("amberfox", reader="root")} == {ids["pub"]}
    assert [m["id"] for m in eng.list(reader="@public")] == [ids["pub"]]


def test_hidden_id_reads_like_a_missing_one(eng):
    ids = seed(eng)
    for reader in ("@public", "@community"):
        assert eng.recall(ids["chef"], reader=reader) is None
        assert eng.recall("00000000-missing", reader=reader) is None
        assert eng.history(ids["chef"], reader=reader) == []
        assert eng.history("00000000-missing", reader=reader) == []
        assert eng.reader_can_see(ids["chef"], reader=reader) is False
        assert eng.reader_can_see("00000000-missing", reader=reader) is False
    assert eng.recall(ids["chef"]) is not None


def test_ceiling_cannot_be_widened_in_process(eng, monkeypatch):
    ids = seed(eng)
    monkeypatch.setenv("SKMEMORY_READER", "@public")
    ro = Engine(eng.root)
    for wide in (None, "@chef-only", "owner", "@chef", "@inner-circle", "@work:x", "channel:x"):
        got = {h["id"] for h in ro.search("amberfox", reader=wide)}
        assert got == {ids["pub"]}, wide
        assert {m["id"] for m in ro.list(reader=wide)} == {ids["pub"]}, wide
        assert ro.recall(ids["chef"], reader=wide) is None, wide
        assert ro.recall(ids["work"], reader=wide) is None, wide
    # a config that makes channel:c1 trusted cannot widen the process ceiling
    cfg = os.path.join(os.path.dirname(eng.root), "aud.json")
    with open(cfg, "w") as f:
        json.dump(
            {
                "channels": {"c1": {"context_tag": "@chef-only", "members": ["A"]}},
                "people": {"A": {"trust_level": 4}},
            },
            f,
        )
    wide = Engine(eng.root, audience_config=cfg)
    assert {h["id"] for h in wide.search("amberfox", reader="channel:c1")} == {ids["pub"]}


@pytest.mark.parametrize("name,args", OWNER_ONLY)
def test_owner_only_raises_permission_error_under_ceiling(eng, monkeypatch, name, args):
    seed(eng)
    monkeypatch.setenv("SKMEMORY_READER", "@public")
    ro = Engine(eng.root)
    with pytest.raises(PermissionError):
        getattr(ro, name)(*args)


def test_python_cannot_obtain_owner_access_under_reader_in_a_fresh_process(eng):
    """The variable is set before the import: nothing the bindings expose gets owner access."""
    ids = seed(eng)
    code = f"""
import json, sys
import skmem_core
from skmem_core import Engine
e = Engine({eng.root!r})
out = {{"hits": sorted(h["id"] for h in e.search("amberfox", reader="@chef-only")),
        "recall_chef": e.recall({ids["chef"]!r}, reader="owner"),
        "denied": []}}
for name, args in {OWNER_ONLY!r}:
    try:
        getattr(e, name)(*args)
    except PermissionError:
        out["denied"].append(name)
out["names"] = [n for n in dir(skmem_core._skmem_core) + dir(Engine) if "scope" in n.lower() or "owner" in n.lower()]
print(json.dumps(out))
"""
    env = dict(os.environ, SKMEMORY_READER="@public")
    r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["hits"] == [ids["pub"]]
    assert out["recall_chef"] is None
    assert out["denied"] == [n for n, _ in OWNER_ONLY]
    assert out["names"] == []
    # nothing was written by the denied calls
    assert len(eng.list()) == 3


def test_reader_process_does_not_create_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SKMEMORY_READER", "@public")
    home = tmp_path / "fresh" / "memory"
    home.mkdir(parents=True)
    Engine(str(home))
    assert not (home / "short-term").exists()


def test_health_reader_gets_verdict_only(eng, monkeypatch):
    seed(eng)
    full = eng.health()
    assert full["primary"]["total_memories"] == 3 and full["status"] == "ok"
    assert set(eng.health(reader="@public")) == {"engine", "ok", "status"}
    monkeypatch.setenv("SKMEMORY_READER", "@public")
    ro = Engine(eng.root)
    assert set(ro.health()) == {"engine", "ok", "status"}
    assert set(ro.health(reader="owner")) == {"engine", "ok", "status"}


def test_recall_reinforces_and_flags_tampering(eng, home):
    ids = seed(eng)
    path = home / "short-term" / f"{ids['pub']}.json"
    doc = json.loads(path.read_text())
    doc["content"] = "tampered"
    path.write_text(json.dumps(doc))
    m = eng.recall(ids["pub"])
    assert "integrity_warning" in m["metadata"]


def test_history_chain_and_hidden_neighbour(eng):
    old = eng.snapshot(
        "v1", "first", context_tag="@public", valid_to="2026-01-01T00:00:00+00:00"
    )
    new = eng.snapshot("v2", "second", context_tag="@public", supersedes_id=old["id"])
    chain = eng.history(new["id"])
    assert [r["id"] for r in chain] == [old["id"], new["id"]]
    assert chain[0]["superseded_by"] == new["id"] and chain[0]["current"] is False
    assert chain[1]["current"] is True and chain[1]["requested"] is True
    hit = [h for h in eng.search("first second") if h["id"] == old["id"]][0]
    assert hit["superseded"] is True and hit["superseded_by"] == new["id"]
    # a hidden successor is not named to a reader, and the closed row looks open
    hid = eng.snapshot("v3", "third", context_tag="@chef-only", supersedes_id=new["id"])
    rows = eng.history(new["id"], reader="@public")
    assert [r["id"] for r in rows] == [old["id"], new["id"]]
    assert rows[-1]["current"] is True and rows[-1]["valid_to"] is None
    assert hid["id"] not in json.dumps(rows)


def test_detail_index_rows(eng):
    seed(eng)
    rows = eng.search("amberfox", detail="index")
    assert rows and set(rows[0]) == {
        "id", "title", "date", "layer", "score", "score_kind", "est_basis", "snippet",
        "source_ref", "expandable", "superseded", "superseded_by", "est_tokens",
        "est_tokens_full",
    }  # fmt: skip
    assert rows[0]["score"] == 1.0 and rows[1]["score"] == 0.5
    assert rows[0]["est_tokens"] < rows[0]["est_tokens_full"]
    with pytest.raises(ValueError):
        eng.search("amberfox", detail="nope")


def test_delete_archive_reindex_roundtrip(eng, home):
    ids = seed(eng)
    assert eng.delete(ids["pub"], reason="test") is True
    assert eng.delete(ids["pub"]) is False
    assert eng.recall(ids["pub"]) is None
    assert (home / "tombstones" / f"{ids['pub']}.json").exists()
    dest = eng.archive(ids["work"])
    assert dest and os.path.exists(dest)
    assert eng.archive("nope") is None
    assert {h["id"] for h in eng.search("amberfox")} == {ids["chef"]}
    rep = eng.reindex(force=True)
    assert rep["indexed"] == 1 and rep["unloadable"] == 0
    assert eng.health()["status"] == "ok"


def test_save_roundtrip_of_a_model_dump(eng):
    m = eng.snapshot("a", "b")
    again = dict(m, id="11111111-1111-4111-8111-111111111111", title="copy")
    assert eng.save(again) == again["id"]
    assert eng.recall(again["id"])["title"] == "copy"


def test_intentions_and_ledger(eng):
    r = eng.remind("  check   the deploy ", "in 2 days", origin_memory_id=None)
    assert r["state"] == "pending" and len(r["id"]) == 16
    listed = eng.intentions()
    assert [i["title"] for i in listed] == ["check the deploy"]
    assert eng.intentions(state="done") == []
    assert eng.intention_done(r["id"])["state"] == "done"
    assert eng.intention_done("nope") is None
    with pytest.raises(ValueError):
        eng.remind("", "in 2 days")
    with pytest.raises(ValueError):
        eng.remind("x", "in 2 days", interval_s=5)
    with pytest.raises(ValueError):
        eng.remind("x", "whenever")
    eid = eng.ledger_record("skill-a", "1", "served", session="s1", host="h")
    assert eid
    eng.ledger_record("skill-a", "1", "followed", effect="success", session="s1", host="h")
    top = eng.ledger_top(host="h")
    assert top[0]["key"] == "skill-a@1" and top[0]["served"] == 1
    with pytest.raises(ValueError):
        eng.ledger_record("skill-a", "1", "bogus")
