# Vendor the BiLLM compatibility sources instead of linking

The runtime-needed BiLLM files (`datautils.py`, `eval_ppl_utils.py`,
`modelutils.py`) were previously symlinked from an external checkout via
`scripts/setup_links.sh`; we now vendor them unmodified under
`external/BiLLM/` so the repository is self-contained and reproducible
without an external checkout, at the cost of manual upstream sync. BiLLM is
MIT-licensed; provenance and attribution are recorded in THIRD_PARTY.md.

Considered: keeping the symlink approach (zero in-repo third-party code, easy
upstream updates) versus vendoring (self-contained, exact-version pinning).
Vendoring won because every experiment must reproduce without the
machine-local BiLLM checkout. One deliberate deviation from upstream is
kept: `llama_eval`/`opt_eval` return the computed perplexity (upstream
returns `None`), which the compatibility boundary depends on.
