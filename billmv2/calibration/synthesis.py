"""Optional local candidate synthesis."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPTS = (
    "Explain the key idea in",
    "A concise technical example of",
    "Compare two approaches to",
    "The following evidence suggests",
)


def synthesize_candidates(
    model: torch.nn.Module,
    tokenizer: object,
    num_samples: int,
    seed_prompts: Sequence[str] = DEFAULT_PROMPTS,
    seed: int = 0,
) -> list[str]:
    """Generate a small local candidate set; failures are non-fatal."""

    if num_samples <= 0:
        return []
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    generated: list[str] = []
    try:
        for index in range(num_samples):
            encoded = tokenizer(seed_prompts[index % len(seed_prompts)], return_tensors="pt")
            input_ids = encoded.input_ids.to(next(model.parameters()).device)
            output = model.generate(
                input_ids,
                max_new_tokens=48,
                do_sample=True,
                generator=generator,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            generated.append(tokenizer.decode(output[0], skip_special_tokens=True))
    except (RuntimeError, ValueError, AttributeError) as error:
        LOGGER.warning("local calibration synthesis failed: %s", error)
    return generated
