"""Numeric-only checks: what a weight file's numbers can reveal about intent.

Read the limits first. A numeric backdoor (a trigger that flips behaviour) is
not decidable from weights alone: the poisoned model and the clean model are
the same kind of object, and no local test proves absence. What these checks
do is make the *shape* of a tampered update visible, using the fact that a
deliberate implant is not distributed like ordinary training noise:

  * LoRA: the update DW = B@A per module has rank <= r. Look at where its
    energy sits. Ordinary training spreads energy across all r directions;
    an implant that must survive quantization concentrates it.

  * Baseline differential: if the pre-finetune base model is on hand, compare
    tensor by tensor. LoRA never touches norms or biases, so those must be
    byte-identical. Any projection that is identical in an otherwise changed
    model, or a norm that is NOT identical, means something other than a plain
    fine-tune happened.

  * Embedding scale outliers: an amplified rare-token row is the cheapest way
    to plant a trigger. Per-block scales can be read without dequantizing.

Every function returns the evidence and names what it cannot prove.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field

from . import _os as _pkg  # noqa: F401  (sets the BLAS thread limits first)
import numpy as np


# --------------------------------------------------------------------------- #
# safetensors / LoRA
# --------------------------------------------------------------------------- #

class _Reader:
    """Sequential safetensors reader. Never holds more than one tensor."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.header = json.loads(self.fh.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def get(self, key):
        info = self.header[key]
        a, b = info["data_offsets"]
        if self.base + b > self.size:
            raise ValueError(f"{key}: truncated (needs {b}, has {self.size - self.base})")
        dt = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
              "I8": np.int8, "I32": np.int32}.get(info["dtype"])
        if dt is None:
            raise ValueError(f"{key}: dtype {info['dtype']} not handled")
        self.fh.seek(self.base + a)
        buf = self.fh.read(b - a)
        return np.frombuffer(buf, dtype=dt).reshape(info["shape"]).astype(np.float32)

    def close(self):
        self.fh.close()


def _sqrtm_psd(G):
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def lora_module_stats(A, B):
    """Spectral facts about DW = B @ A without ever forming DW.

    DW is m x n; A is r x n, B is m x r. With G = A A^T (r x r),
    DW DW^T = B G B^T, whose nonzero eigenvalues equal those of
    G^(1/2) (B^T B) G^(1/2) -- an r x r problem. Cost is O(m r^2), not O(m n r).
    """
    G = A @ A.T
    H = B.T @ B
    S = _sqrtm_psd(G)
    M = S.T @ H @ S
    M = (M + M.T) / 2.0
    lam = np.linalg.eigvalsh(M)
    lam = np.clip(lam, 0.0, None)
    lam = np.sort(lam)[::-1]
    tot = float(lam.sum())
    if tot <= 0:
        return dict(energy=0.0, top1=0.0, top2=0.0, eff_rank=0.0, fro=0.0,
                    max_b=float(np.abs(B).max()), max_a=float(np.abs(A).max()))
    p = lam / tot
    nz = p[p > 0]
    return dict(
        energy=float(np.sqrt(tot)),                      # ||DW||_F
        top1=float(p[0]),
        top2=float(p[0] + (p[1] if p.size > 1 else 0.0)),
        eff_rank=float(np.exp(-(nz * np.log(nz)).sum())),
        fro=float(np.sqrt(tot)),
        max_b=float(np.abs(B).max()),
        max_a=float(np.abs(A).max()),
    )


def _pairs(header):
    mods = {}
    for k in header:
        if ".lora_A" in k or ".lora_B" in k:
            parts = k.split(".")
            i = [j for j, x in enumerate(parts) if x.startswith("lora_")][0]
            m = ".".join(parts[:i])
            mods.setdefault(m, {})["A" if "lora_A" in parts[i] else "B"] = k
    return {m: v for m, v in mods.items() if "A" in v and "B" in v}


def _mad_z(values):
    v = np.asarray(values, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-12
    return (v - med) / (1.4826 * mad), med, mad


@dataclass
class NumericReport:
    path: str
    kind: str
    verdict: str = "clean"
    findings: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    cannot_prove: str = (
        "This compares the shape of the update, not its meaning. A backdoor "
        "distributed like ordinary training noise is indistinguishable here, "
        "and absence of an anomaly is not absence of a trigger."
    )

    def as_dict(self):
        return {"path": self.path, "kind": self.kind, "verdict": self.verdict,
                "findings": self.findings, "evidence": self.evidence,
                "cannot_prove": self.cannot_prove}


def audit_lora(path, z_threshold=6.0, top1_threshold=0.75):
    """Rank/energy structure of every LoRA module in one adapter file."""
    rep = NumericReport(path=path, kind="lora")
    r = _Reader(path)
    try:
        pairs = _pairs(r.header)
        rows = []
        for name, v in pairs.items():
            try:
                A, B = r.get(v["A"]), r.get(v["B"])
            except ValueError as exc:
                rep.findings.append(f"{name}: {exc}")
                rep.verdict = "dangerous"
                continue
            st = lora_module_stats(A, B)
            st["module"] = name
            st["rank"] = int(min(A.shape[0], B.shape[1]))
            rows.append(st)
    finally:
        r.close()

    if not rows:
        rep.verdict = "review"
        rep.findings.append("no LoRA A/B pairs found")
        return rep

    energies = np.array([x["energy"] for x in rows])
    maxb = np.array([x["max_b"] for x in rows])
    top1 = np.array([x["top1"] for x in rows])
    effr = np.array([x["eff_rank"] for x in rows])
    z_e, med_e, _ = _mad_z(energies)
    z_b, med_b, _ = _mad_z(maxb)
    z_r, med_r, _ = _mad_z(effr)
    med_t1 = float(np.median(top1))

    # A robust z alone is not evidence: when every module is within 1% of the
    # median, the MAD collapses and a 0.0001 difference becomes z=-13. An
    # implant has to move the number, not just stand out from a flat cohort.
    # So every flag needs BOTH a z score and a ratio against the median.
    def rel(v, m):
        return float(v / m) if m else float("inf")

    for i, x in enumerate(rows):
        short = x["module"].split("base_model.model.", 1)[-1]
        rb, re_, rr = rel(maxb[i], med_b), rel(energies[i], med_e), rel(effr[i], med_r)

        if x["max_b"] == 0.0:
            rep.findings.append(f"{short}: lora_B is exactly zero (untrained, cleared, or never written)")
            rep.verdict = "dangerous"
            continue

        if abs(z_b[i]) > z_threshold and (rb > 2.0 or rb < 0.5):
            rep.findings.append(
                f"{short}: |B| is {rb:.2f}x the median ({x['max_b']:.3e} vs {med_b:.3e})")
            if rep.verdict == "clean":
                rep.verdict = "review"

        if x["top1"] > 0.90 and rel(x["top1"], med_t1) > 1.5:
            rep.findings.append(
                f"{short}: {x['top1']*100:.0f}% of the update energy in one direction "
                f"(median module: {med_t1*100:.0f}%)")
            if rep.verdict == "clean":
                rep.verdict = "review"

        if abs(z_r[i]) > z_threshold and rr < 0.5:
            rep.findings.append(
                f"{short}: effective rank {x['eff_rank']:.2f} vs median {med_r:.2f} "
                "-- the update collapsed onto few directions")
            if rep.verdict == "clean":
                rep.verdict = "review"

        if abs(z_e[i]) > z_threshold and (re_ > 3.0 or re_ < 0.33):
            rep.findings.append(
                f"{short}: ||dW||_F is {re_:.2f}x the median ({x['energy']:.3e} vs {med_e:.3e})")
            if rep.verdict == "clean":
                rep.verdict = "review"

    truncated = [f for f in rep.findings if "truncated" in f]
    if len(truncated) > 5:
        rep.findings = [f for f in rep.findings if "truncated" not in f]
        rep.findings.insert(0, f"{len(truncated)} tensors are truncated -- the file is incomplete")
        rep.verdict = "dangerous"

    rep.evidence = {
        "modules": len(rows),
        "bad_modules": sum(1 for x in rows if x["max_b"] == 0.0),
        "rank": sorted({x["rank"] for x in rows}),
        "energy_median": float(med_e), "energy_min": float(energies.min()),
        "energy_max": float(energies.max()),
        "max_b_median": float(med_b), "max_b_min": float(maxb.min()),
        "max_b_max": float(maxb.max()),
        "top1_median": med_t1, "top1_max": float(top1.max()),
        "eff_rank_median": float(med_r), "eff_rank_min": float(effr.min()),
        "cohort_cv_max_b": float(maxb.std() / med_b) if med_b else None,
    }
    return rep


# --------------------------------------------------------------------------- #
# GGUF baseline differential
# --------------------------------------------------------------------------- #

GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
             8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
             13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
             18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
             22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
             27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}


def gguf_index(path):
    """Tensor table only: name -> (dims, type, absolute offset, byte length)."""
    fh = open(path, "rb")
    try:
        if fh.read(4) != b"GGUF":
            raise ValueError("not GGUF")
        fh.read(4)
        n_t, n_kv = struct.unpack("<QQ", fh.read(16))
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        align = 32

        def rstr():
            (ln,) = struct.unpack("<Q", fh.read(8))
            return fh.read(ln).decode("utf-8", "replace")

        def rval(t, d=0):
            if t == 8:
                return rstr()
            if t == 9 and d < 2:
                (et,) = struct.unpack("<I", fh.read(4))
                (cnt,) = struct.unpack("<Q", fh.read(8))
                vals = [rval(et, d + 1) for _ in range(min(cnt, 4))]
                if cnt > 4:
                    fh.read(0)
                    for _ in range(cnt - 4):
                        rval(et, d + 1)
                return vals
            fh.read(sizes.get(t, 4))
            return None

        kv = {}
        for _ in range(n_kv):
            k = rstr()
            (t,) = struct.unpack("<I", fh.read(4))
            v = rval(t)
            if k in ("general.alignment", "general.name", "general.finetune",
                     "general.basename", "general.size_label", "general.quantization_version"):
                kv[k] = v
        if isinstance(kv.get("general.alignment"), int):
            align = kv["general.alignment"]
        tens = {}
        for _ in range(n_t):
            nm = rstr()
            (nd,) = struct.unpack("<I", fh.read(4))
            dims = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
            (tt,) = struct.unpack("<I", fh.read(4))
            (off,) = struct.unpack("<Q", fh.read(8))
            tens[nm] = [dims, tt, off]
        pos = (fh.tell() + align - 1) // align * align
    finally:
        fh.close()
    order = sorted(tens.items(), key=lambda x: x[1][2])
    out = {}
    for i, (nm, t) in enumerate(order):
        end = order[i + 1][1][2] if i + 1 < len(order) else os.path.getsize(path) - pos
        out[nm] = dict(dims=t[0], gtype=t[1], start=pos + t[2], nbytes=max(0, end - t[2]))
    return out, kv, pos


def _kind(nm):
    if "norm" in nm:
        return "norm"
    if nm.endswith("bias"):
        return "bias"
    if "token_embd" in nm:
        return "embed"
    if nm.startswith("output"):
        return "output"
    if "ffn_" in nm:
        return "ffn_proj"
    if "attn_" in nm:
        return "attn_proj"
    return "other"


def _compare_tensor(fa, fb, a, b, block=1 << 20, rel_tol=1e-6):
    """Same tensor in two GGUFs -> 'same' | 'noise' | 'changed' | 'truncated'.

    Unquantized tensors (F32/F16/BF16) get a numeric comparison with a ULP
    allowance, because two conversion runs of the same checkpoint can differ in
    the last bit. Quantized tensors get a byte comparison: their encoding is
    deterministic, so any difference is a real difference in the weights.
    """
    if tuple(a["dims"]) != tuple(b["dims"]):
        return "shape_mismatch", None
    # Different quantization type = different byte length by design. Comparing
    # them byte-wise would be meaningless, and calling it tampering is worse:
    # a finetune re-quantized with its own recipe legitimately arrives as Q6_K
    # where the base was Q4_K.
    if a["gtype"] != b["gtype"]:
        return "requantized", None
    # The last tensor's length is measured against end-of-file, so a header that
    # differs in size shifts it by a few bytes. Anything within one alignment
    # page is padding, not a missing tail.
    if abs(a["nbytes"] - b["nbytes"]) > 1024:
        return "truncated", None
    n = min(a["nbytes"], b["nbytes"])
    num_dtype = {0: np.float32, 1: np.float16, 30: None}.get(a["gtype"], "byte")
    pos = 0
    worst = 0.0
    while pos < n:
        ln = min(block, n - pos)
        fa.seek(a["start"] + pos); ca = fa.read(ln)
        fb.seek(b["start"] + pos); cb = fb.read(ln)
        if ca != cb:
            if num_dtype == "byte" or num_dtype is None:
                return "changed", None
            ln2 = ln - (ln % 4 if num_dtype is np.float32 else ln % 2)
            if ln2 == 0:
                return "changed", None
            va = np.frombuffer(ca[:ln2], dtype=num_dtype).astype(np.float32)
            vb = np.frombuffer(cb[:ln2], dtype=num_dtype).astype(np.float32)
            d = np.abs(va - vb)
            scale = np.abs(vb).max() or 1.0
            worst = max(worst, float(d.max() / scale))
            if worst > rel_tol:
                return "changed", worst
        pos += ln
    return ("noise" if worst > 0 else "same"), (worst or None)


def gguf_baseline_diff(model_path, base_path, rel_tol=1e-6):
    """Compare a finetuned GGUF against its claimed base, tensor by tensor.

    The output is a *pattern*, which is the only thing weights can tell you:
    what changed tells you what kind of update happened. A LoRA merge adapts
    projections and leaves norms and biases alone; if those moved, the story
    told by the metadata is not the story in the numbers.
    """
    rep = NumericReport(path=model_path, kind="gguf-vs-base")
    ta, kva, _ = gguf_index(model_path)
    tb, kvb, _ = gguf_index(base_path)
    common = [n for n in ta if n in tb]
    if not common:
        rep.verdict = "review"
        rep.findings.append("no tensor names in common: not the same architecture")
        return rep

    verdicts = {}
    fa, fb = open(model_path, "rb"), open(base_path, "rb")
    try:
        for nm in common:
            verdicts[nm] = _compare_tensor(fa, fb, ta[nm], tb[nm], rel_tol=rel_tol)
    finally:
        fa.close(); fb.close()

    counts = {}
    by_kind = {}
    for nm, (v, _w) in verdicts.items():
        counts[v] = counts.get(v, 0) + 1
        k = _kind(nm)
        by_kind.setdefault(k, {}).setdefault(v, 0)
        by_kind[k][v] += 1

    changed = [nm for nm, (v, _w) in verdicts.items() if v == "changed"]
    noise = [nm for nm, (v, _w) in verdicts.items() if v == "noise"]
    untouched_norm = by_kind.get("norm", {}).get("same", 0) + by_kind.get("norm", {}).get("noise", 0)
    changed_norm = by_kind.get("norm", {}).get("changed", 0)
    changed_bias = by_kind.get("bias", {}).get("changed", 0)

    if changed_norm or changed_bias:
        examples = [n for n in changed if _kind(n) in ("norm", "bias")][:4]
        rep.findings.append(
            f"{changed_norm} norm and {changed_bias} bias tensors differ by more than float "
            f"noise from the claimed base ({', '.join(examples)}). A LoRA merge leaves both "
            "untouched -- they are not adapted -- so either the base in the metadata is not "
            "the base used, or the merge was not plain LoRA")
        rep.verdict = "review"

    unchanged_heavy = [nm for nm in common
                       if verdicts[nm][0] in ("same", "noise") and ta[nm]["nbytes"] > (1 << 20)]
    if unchanged_heavy and changed:
        rep.findings.append(
            f"{len(unchanged_heavy)} large weight tensors are identical to the base in a model "
            f"whose other {len(changed)} tensors changed (e.g. {unchanged_heavy[0]}). "
            "That is a partial update, not a full finetune -- worth explaining")
        if rep.verdict == "clean":
            rep.verdict = "review"

    if noise and not rep.findings:
        rep.findings.append(
            f"{len(noise)} tensors differ only in the last bits of float values "
            "(a re-save, not an edit)")

    # The pattern itself is the evidence: say what it looks like.
    if changed and not changed_norm and not changed_bias:
        rep.findings.append(
            "pattern is consistent with a LoRA merge: every projection changed, every norm "
            "and bias identical")

    requant = [nm for nm, (v, _w) in verdicts.items() if v == "requantized"]
    if requant:
        pairs = sorted({f"{GGML_TYPE.get(ta[n]['gtype'], ta[n]['gtype'])}->"
                        f"{GGML_TYPE.get(tb[n]['gtype'], tb[n]['gtype'])}" for n in requant})
        rep.findings.append(
            f"{len(requant)} tensors use a different quantization than the base "
            f"({', '.join(pairs)}): the finetune was re-quantized with its own recipe, "
            "so those tensors cannot be compared byte-wise")

    shape_bad = [nm for nm, (v, _w) in verdicts.items() if v == "shape_mismatch"]
    if shape_bad:
        rep.findings.append(
            f"{len(shape_bad)} tensors have a different shape than the base "
            f"({', '.join(shape_bad[:4])}) -- this is not the same architecture")
        rep.verdict = "dangerous"

    trunc = counts.get("truncated", 0)
    if trunc:
        rep.findings.append(
            f"{trunc} tensors have the same quantization but a different byte length "
            "-- those tensors are not the ones in the base")
        if rep.verdict == "clean":
            rep.verdict = "review"

    rep.evidence = {
        "tensors_compared": len(common),
        "same": counts.get("same", 0), "float_noise_only": counts.get("noise", 0),
        "changed": counts.get("changed", 0),
        "requantized": counts.get("requantized", 0),
        "shape_mismatch": counts.get("shape_mismatch", 0),
        "truncated": counts.get("truncated", 0),
        "by_kind": by_kind,
        "unchanged_norms_and_biases": untouched_norm,
        "model_meta": {k: kva.get(k) for k in ("general.name", "general.finetune",
                                               "general.basename", "general.size_label")},
        "base_meta": {k: kvb.get(k) for k in ("general.name", "general.finetune",
                                              "general.basename", "general.size_label")},
    }
    return rep


# --------------------------------------------------------------------------- #
# GGUF embedding outlier scan (scale proxy, no dequantization)
# --------------------------------------------------------------------------- #

def _embedding_scales(path, samples=0):
    """Per-token-row max block scale of token_embd, without dequantizing."""
    idx, _kv, _ = gguf_index(path)
    name = next((n for n in idx if "token_embd.weight" in n), None)
    if name is None:
        return None, "no token_embd.weight in this GGUF"
    t = idx[name]
    gtype = t["gtype"]
    if gtype not in (12, 13, 14):
        return None, f"embedding is {GGML_TYPE.get(gtype, gtype)}, not a K-quant"
    blocksize = {12: 144, 13: 176, 14: 210}[gtype]
    per = 256
    ncols = int(t["dims"][0]) if len(t["dims"]) == 2 else 0
    if ncols % per:
        return None, "embedding row is not a multiple of 256"
    nb = ncols // per
    rowbytes = nb * blocksize
    nrows = t["nbytes"] // rowbytes
    if nrows == 0:
        return None, "embedding tensor too small to scan"
    scales = np.empty(nrows, dtype=np.float32)
    fh = open(path, "rb")
    try:
        step = 4096
        for r0 in range(0, nrows, step):
            r1 = min(nrows, r0 + step)
            fh.seek(t["start"] + r0 * rowbytes)
            raw = np.frombuffer(fh.read((r1 - r0) * rowbytes), dtype=np.uint8)
            blk = raw.reshape(r1 - r0, nb, blocksize)
            if gtype in (12, 13):
                d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)
            else:                        # Q6_K puts the fp16 scale at the block tail
                d = blk[:, :, blocksize - 2:blocksize].copy().view(np.float16).astype(np.float32)
            scales[r0:r1] = np.abs(d).reshape(r1 - r0, nb).max(axis=1)
    finally:
        fh.close()
    return (scales, dict(tensor=name, rows=int(nrows), blocks_per_row=int(nb))), None


def gguf_embedding_outliers(path, base_path=None, top=8, ratio_threshold=3.0):
    """Per-token-row embedding scale comparison.

    A row whose block scale is far above its cohort looks like an amplified
    rare token -- but a clean Q4_K embedding has such rows too (the base model
    in this repo's example has a row at z=88). Standing out is therefore not
    evidence. Only a *change against the base* is, so without a base this
    check reports informationally and never raises the verdict.
    """
    rep = NumericReport(path=path, kind="gguf-embedding")
    got, why = _embedding_scales(path)
    if got is None:
        rep.verdict = "review"
        rep.findings.append(why)
        return rep
    scales, info = got
    med = float(np.median(scales))
    hi = np.argsort(-scales)[:top]
    rep.evidence = dict(info, scale_median=med,
                        scale_p95=float(np.percentile(scales, 95)),
                        scale_max=float(scales.max()),
                        top_rows=[(int(i), float(scales[i])) for i in hi])

    if not base_path:
        rep.findings.append(
            f"{len(scales)} token rows scanned; {len(hi)} largest scales listed. "
            "Scale outliers occur in clean quantizations too, so this alone is "
            "informational -- pass --base to compare against the original")
        return rep

    bgot, _bwhy = _embedding_scales(base_path)
    bscales = bgot[0] if bgot else None
    if bscales is None or bscales.shape != scales.shape:
        rep.verdict = "review"
        rep.findings.append("base embedding not comparable (shape or format differs)")
        return rep

    ratio = scales / np.maximum(bscales, 1e-12)
    hot = np.argsort(-ratio)[:top]
    flagged = [(int(i), float(ratio[i]), float(scales[i]), float(bscales[i]))
               for i in hot if ratio[i] > ratio_threshold]
    rep.evidence["ratio_max"] = float(ratio.max())
    rep.evidence["rows_amplified_over_%.1fx" % ratio_threshold] = int((ratio > ratio_threshold).sum())
    for i, r_, a_, b_ in flagged:
        rep.findings.append(
            f"token row {i}: scale {a_:.4e} vs base {b_:.4e} = {r_:.1f}x "
            "-- an amplified embedding row, the cheapest trigger implant")
    if flagged:
        rep.verdict = "review"
    return rep
