# BiLLMv2 Context

BiLLMv2 is a post-training quantization (PTQ) system for decoder-only language
models, built on BiLLM's structured binary representation. It quantizes weights
layer by layer into compact, reloadable artifacts and evaluates them through a
compatibility boundary over vendored BiLLM sources.

## Language

### Quantization

**Artifact**:
A compact, reloadable representation of a quantized model: packed binary
payloads, low-rank factors, rotations, scales, configuration, and metrics —
stored separately from the original model weights.
_Avoid_: checkpoint, model dump

**Split**:
The structural division of a weight matrix into salient and non-salient
regions, each quantized with a different strategy.
_Avoid_: split ratio, mask

**Salient columns**:
The columns within each block that are treated at higher precision than the
bulk binary-quantized columns.

**Low-rank compensation**:
A rank-r factor branch (functional branch) that adds a correction to a
quantized layer's output to recover quantization error.

**Joint search**:
A search that jointly decides rotation, salient-column count, and low-rank
rank within a shared bit budget.

**BPW (bit-per-weight)**:
The number of bits spent per weight parameter; the unit of the bit budget.
_Avoid_: bits, size

### Calibration

**Calibration data**:
Token windows used for layer-by-layer reconstruction. Not training data —
weights are never updated, only re-expressed.

**Candidate pool**:
The full set of token windows drawn from a calibration dataset, larger than
what is actually used, from which a representative subset is selected.
_Avoid_: calibration set, pool samples

**Calibration selection**:
Choosing a representative subset of the candidate pool (k-center, D-optimal,
or hybrid).

**Token cache**:
A persisted, tokenized form of a dataset; uniquely keyed by (dataset,
nsamples, seed, seqlen, model).

### Entry points

**PTQ**:
The main entry (`run_ptq.py`): a gradient-free, layer-by-layer quantization
pipeline that persists artifacts.
_Avoid_: main script

**PTQ+**:
The refinement entry (`run_ptq_ft.py`): calibration-only fine-tuning after
PTQ, training only rotation, scales, and low-rank parameters.

**Evaluator**:
The read-only entry (`evaluate.py`) that loads artifacts, reapplies them to
original weights, and computes perplexity.

**Compatibility boundary**:
The import boundary (`baseline.py`, `calibration/data.py`) through which
vendored BiLLM sources — layer discovery, perplexity evaluation, and dataset
loading — are reached.

### Environment

**Workspace**:
The machine-local volume hosting HF model/dataset caches, token caches, and
run outputs (`/autodl-fs/data/cclanro`); never inside the repository.

**Perplexity (PPL)**:
The evaluation metric reported for quantized and baseline models.
_Avoid_: loss, error

## Mechanism

PTQ re-expresses each linear layer by splitting weight columns into salient
and non-salient regions, binary-quantizing the bulk, and optionally attaching
a low-rank compensation branch; a joint search picks salient-column counts
and ranks under a fixed BPW budget. The resulting artifacts can be reloaded
onto original weights and evaluated for perplexity through the compatibility
boundary.

## Results

All runs on `meta-llama/Meta-Llama-3-8B`, wikitext2.

| Experiment | PPL | Provenance |
|---|---|---|
| FP16 baseline | 6.138 | logs/wikitext2_baseline.log |
| Minimal PTQ (binary + GPTQ, no low-rank) | 15.97 | logs/ptq_8b_min.log |
| F2 preset (joint search + low-rank) | 81.57 | logs/ptq_8b.log |
| F2 preset, reloaded | 81.57 | logs/eval_reload.log |
| F2 preset minus low-rank adapters | 21.37 | logs/eval_nolr.log |

Key finding: on this model the low-rank correction is orthogonal to the
measured quantization residual (cosine ≈ 0, logs/lr_diag.log), so the F2
low-rank compensation hurts rather than helps. The minimal configuration is
the only trustworthy result so far.
