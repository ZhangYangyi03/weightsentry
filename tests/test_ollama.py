"""Ollama integrity: a flipped byte in a blob must be caught by re-hashing."""
import hashlib
import json
import os

from weightsentry import ollama


def build_store(root, payload=b"weights-bytes" * 100):
    blobs = os.path.join(root, "blobs")
    man = os.path.join(root, "manifests", "registry.ollama.ai", "library", "tiny")
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(man, exist_ok=True)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    with open(os.path.join(blobs, digest.replace(":", "-")), "wb") as fh:
        fh.write(payload)
    spec = {"layers": [{"mediaType": "application/vnd.ollama.image.model",
                        "digest": digest, "size": len(payload)}]}
    with open(os.path.join(man, "1b"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return root, digest, len(payload)


def test_intact_store_is_clean(tmp_path):
    root, _d, _n = build_store(str(tmp_path))
    models = ollama.list_models([root])
    assert len(models) == 1
    m = ollama.verify_model(models[0], root)
    assert m.verdict() == "clean", m.errors
    assert m.blobs[0].sha_ok is True


def test_tampered_blob_is_dangerous(tmp_path):
    root, digest, _n = build_store(str(tmp_path))
    p = os.path.join(root, "blobs", digest.replace(":", "-"))
    with open(p, "ab") as fh:
        fh.write(b"\x00")          # one byte appended
    m = ollama.verify_model(ollama.list_models([root])[0], root)
    assert m.verdict() == "dangerous"
    assert m.blobs[0].sha_ok is False


def test_missing_blob_is_dangerous(tmp_path):
    root, digest, _n = build_store(str(tmp_path))
    os.remove(os.path.join(root, "blobs", digest.replace(":", "-")))
    m = ollama.verify_model(ollama.list_models([root])[0], root)
    assert m.verdict() == "dangerous"


def test_zero_filled_blob_is_named_as_such(tmp_path):
    root, digest, n = build_store(str(tmp_path))
    p = os.path.join(root, "blobs", digest.replace(":", "-"))
    with open(p, "wb") as fh:
        fh.write(b"\x00" * n)          # right size, no content
    m = ollama.verify_model(ollama.list_models([root])[0], root)
    assert m.verdict() == "dangerous"
    assert m.blobs[0].zerofill is True
