#!/usr/bin/env python3
"""Evaluate a model reconstructed from compact BiLLM-v2 artifacts."""

from __future__ import annotations

import argparse
import logging

from billmv2.baseline import (
    evaluate_perplexity,
    evaluate_perplexity_limited,
    load_model,
)
from billmv2.calibration.data import get_candidate_pool
from billmv2.utils.artifacts import (
    apply_billmv2_artifacts,
    load_billmv2_artifacts,
)
from billmv2.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load original weights, apply compact artifacts, and evaluate perplexity."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset", choices=["wikitext2", "c4", "ptb"], default="wikitext2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    configure_logging()
    bundle = load_billmv2_artifacts(args.artifact_dir)
    model_name = args.model or bundle["config"]["model"]
    model = load_model(model_name).eval()
    seqlen = int(bundle["config"].get("seqlen", 0))
    if seqlen > 0:
        model.seqlen = seqlen
    apply_billmv2_artifacts(model, bundle)
    _, test_loader = get_candidate_pool(
        args.dataset, 1, int(bundle["config"]["seed"]), model.seqlen, model_name
    )
    ppl = (
        evaluate_perplexity_limited(model, test_loader, args.device, args.max_samples)
        if args.max_samples > 0
        else evaluate_perplexity(model, test_loader, args.device, args.dataset)
    )
    LOGGER.info("artifact PPL (%s): %.6f", args.dataset, ppl)


if __name__ == "__main__":
    main()
