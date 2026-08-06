# Third-party references

No source file from SpinQuant, QuaRot, EoRA, COLA, or Output Alignment is copied
into this repository. The implementations named below were written independently
from the algorithms described by the papers and public project documentation.

| Project | License observed | Files influenced here | Use and changes |
|---|---|---|---|
| BiLLM | MIT | `baseline.py`, `pipeline/sequential.py`, `quantization/*` | `datautils.py`, `eval_ppl_utils.py` and `modelutils.py` are vendored under `external/BiLLM/` (source: `hawkheimmer/BiLLM-branch` @ `2a443a5`, a fork of `htqin/BiLLM`), with one fix: `llama_eval`/`opt_eval` return their computed perplexity (upstream returns `None`). `modelutils.find_layers` and perplexity evaluation are imported through a compatibility boundary. Structured salient/non-salient branches and GPTQ/OBC ordering are retained, while scoring, splitting, scales, and artifact handling are new. |
| QuaRot | Apache-2.0 | `transforms/rotation.py`, `transforms/folding.py` | Algorithmic reference for function-preserving/foldable rotations. The block rotation and folding code is independent and does not copy QuaRot source. |
| SpinQuant | CC-BY-NC-4.0 | `finetune/refinement.py` | Algorithmic reference for Cayley-parameterized learned rotations. No SpinQuant code is copied because of the non-commercial license; the blockwise Cayley implementation is independent. |
| EoRA | NVIDIA Source Code License-NC | `low_rank/compensation.py` | Algorithmic reference for activation-aware low-rank compensation. No EoRA code is copied; weighted truncated SVD is independently implemented. |
| COLA | MIT | `calibration/features.py`, `calibration/selector.py` | Conceptual reference for calibration diversity/representativeness. Feature projection, k-center, rank-one D-optimal update, and hybrid objective are independent. |
| Output Alignment (arXiv:2512.21651) | Paper | `reconstruction/objectives.py`, `pipeline/sequential.py` | Conceptual reference for teacher/student output alignment. The diagonal soft-whitening objective and streaming implementation are independent. |

License sources were checked on 2026-07-29 at each project's public repository.
Model weights and datasets retain their own licenses and are never redistributed.
