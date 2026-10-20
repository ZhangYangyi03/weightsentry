weightsentry
============

Static supply-chain audit for the model artifacts sitting on your own disk.

It answers one question per file: if something loads this, can it run code on
me, and does what is inside match what the header claims? It never loads the
model, never imports torch, and never unpickles a single byte. Everything is
done by reading bytes and walking opcode streams -- which is the only way to
inspect a hostile checkpoint without executing it.

    pip install -e .
    weightsentry ~                          # static audit of everything model-shaped under $HOME
    weightsentry ./models --numeric         # add the numeric checks
    weightsentry ./model.gguf --base ./base.gguf --numeric
    weightsentry ./models --json > report.json

What it checks
--------------

  format          three model containers can execute code by design.
                  weightsentry says which is which instead of calling them
                  all "model files".

                  gguf         inert: tensor + metadata container
                  safetensors  inert: 8-byte header + raw bytes
                  onnx         inert graph; custom op domains are the risk
                  zip-torch    zip whose data.pkl torch.load unpickles
                  pickle       raw pickle: the opcode stream is a program

  pickle RCE      every GLOBAL / STACK_GLOBAL / REDUCE / BUILD / NEWOBJ is
                  extracted and the named callables are classified. os.system,
                  subprocess.Popen, socket, urllib, requests, ctypes,
                  builtins.__import__ and friends are dangerous; an unexpected
                  callable outside the known-benign torch/numpy set is review.
                  Truncated or non-pickle input is review, never a crash.

  safetensors     header JSON, tensor span vs dtype*shape, overlaps, offsets
                  past EOF, bytes of trailing data after the last tensor, and
                  unexpected __metadata__ keys.

  ollama blob     every blob in ~/.ollama/models is re-hashed against the
                  digest its manifest declares. A flipped byte, an appended
                  byte, or a manifest pointing at a blob from elsewhere all
                  show up. Manifest-level mediaType is reported, so a model
                  that ships an adapter layer is visible.

Exit codes
----------

  0  nothing above --fail-on
  2  a finding at or above --fail-on (default: dangerous)

so it drops into CI as a gate.

    weightsentry ~ --fail-on review

Numeric checks (--numeric, --base)
----------------------------------

The static layer answers "can this file execute code". This one answers "do the
numbers look like an ordinary update, or like an implant". It needs numpy and
nothing else, and it never loads a model.

    weightsentry ./models --numeric
    weightsentry ./model.gguf --base ./base.gguf --numeric

What it actually measures, and why that is the most weights can tell you:

  LoRA rank-energy    For each module, the update dW = B@A has rank <= r. The
                      spectrum of dW dW^T is computed from the r x r matrices
                      A A^T and B^T B, so dW is never formed: O(m r^2), one
                      small tensor in memory at a time. Three numbers come out
                      per module -- ||dW||_F, the share of energy in the top
                      direction, and the effective rank. An implant that must
                      survive quantization concentrates energy; ordinary
                      training spreads it.

  GGUF differential   Given the pre-finetune base, every tensor is compared.
                      Unquantized tensors (F32/F16/BF16) get a numeric compare
                      with a float-noise allowance; quantized tensors get byte
                      equality, because their encoding is deterministic. The
                      result is a *pattern*: which kinds of tensor changed.

                      A LoRA merge adapts projections and leaves norms and
                      biases alone. If a norm moved by more than float noise,
                      the base in the metadata is not the base used. If a large
                      weight tensor is byte-identical inside a model whose
                      others changed, the update was partial.

  Embedding scales    Q4_K/Q6_K store one fp16 scale per 256 weights, so a
                      deliberately amplified token row is visible in the scales
                      without dequantizing. This is only reported against a
                      base: clean quantizations have extreme rows too (the base
                      model in examples/ has one at z=88), so an outlier on its
                      own is not evidence and never raises the verdict alone.

Why z-scores alone are not used: when every module sits within 1% of the
cohort median, the MAD collapses and a 0.0001 difference becomes z=-13. Every
flag therefore also has to clear a ratio test against the median -- a flag has
to move the number, not just stand out in a flat distribution. tests/test_numeric.py
holds that case explicitly.

The honest limit
----------------

No local test proves a numeric backdoor is absent. A trigger that is
distributed like ordinary training noise is indistinguishable from training
noise by construction, and this tool will report clean on it. What it can do is
make a deliberate, structured implant visible -- and say precisely which check
passed, so the claim you make afterwards is as narrow as the evidence.

Proving a specific trigger needs a different experiment: hold out the base
model, run clean and trigger-conditioned inputs through both, and compare
outputs. That is behaviour, not weights, and it is out of scope here.

Known limits, stated rather than hidden
---------------------------------------

  * Numeric backdoors are not detectable from weights alone: --numeric narrows
    the space by comparing against a base model, and proves nothing on its own.
    A trigger distributed like training noise reads as clean.
  * Reloading trust is out of scope: it verifies that a blob matches its
    manifest, not that the digest came from the publisher you think it did.
    Pin upstream digests separately.
  * ONNX custom op domains can pull in native code at runtime; weightsentry
    reports the format but does not resolve op domain implementations.
  * GGUF is parsed for header metadata only.

License: Apache-2.0.

A real run, for calibration
---------------------------

On a Windows machine with two Qwen2.5 LoRA adapters, one merged 7B GGUF, two
Silero VAD ONNX models, and five Ollama models, every file was clean and one
finding was real:

    dangerous  safetensors  58.9 MB   .../paper4_CoLoRA/adapters/math_lora/adapter_model.safetensors
               ! tensors run 99765512B past EOF

That file has the same 392-tensor header as its sibling code_lora adapter,
which is 161,533,192 bytes and clean. math_lora is 61,767,680 bytes -- a
truncated write. Nothing was executed; the header arithmetic said so.

And on the Ollama side, three of five models were all-zero placeholders:
qwen2.5:0.5b, qwen2.5:3b and qwen2.5:14b each carry a `model` blob of exactly
the declared size filled entirely with 0x00. They fail their manifest digest,
but they are not tampered with -- they are interrupted pulls. That difference
is why the report says "all-zero placeholder (interrupted pull)" instead of
"tampered": a tool that cries wolf on a bad download teaches people to ignore
it. `ollama rm` + `ollama pull` fixes all three.

The full report is in examples/REPORT_example.json.

The numeric pass on the same tree:

    review    lora  .../lora_adapter/adapter_model.safetensors
              ! model.layers.1.mlp.gate_proj: 97% of the update energy in one direction
                (median module: 53%)          -- 40 modules the same way
    review    lora  .../4_safety_cliff_exp/adapters/output_1000/adapter_model.safetensors
              ! model.layers.1.mlp.gate_proj: 94% ... (median 38%)
    clean     gguf-vs-base  ruzheng-q4.gguf vs the Qwen2.5-7B base
              132 identical, 192 changed, 6 requantized, 9 float-noise only
              ! pattern is consistent with a LoRA merge: every projection changed,
                every norm and bias identical

The energy-collapse finding is what this layer is for: a handful of MLP modules
whose update lives almost entirely in one direction. It is not proof of an
implant -- strongly targeted finetunes look like this too -- it is the list of
places to look with a behaviour test.

The GGUF differential also shows a common false alarm: six tensors arrived
Q6_K where the base was Q4_K. The finetune was re-quantized with its own recipe.
Byte comparison across different quantizations is meaningless, so those are
reported as requantized, not as tampering.

What this run does NOT claim
----------------------------

No numeric backdoor was found, and this tool cannot prove one absent. What the
run establishes is narrower and worth stating exactly: none of these artifacts
can execute code when loaded, the one that is broken is visibly broken, and the
adapters whose update collapsed onto a single direction are named so someone
can test their behaviour rather than their weights.
