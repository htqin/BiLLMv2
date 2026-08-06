# Host large outputs and caches on the machine-local volume

Default run outputs, token caches, and HF model/dataset caches live under
`/autodl-fs/data/cclanro` (`billm-v2-output/`, `huggingface/`) instead of
inside the repository, because that volume is large (14 TB) while the
repository and home volume are small. Repository `.gitignore` keeps the repo
free of models, datasets, caches, and run products.

Constraints not visible in the code: `/autodl-fs` is a machine-local scratch
volume — contents are not durable across machine recreation and must never be
treated as a backup. HF cache paths are honored from environment variables
(`HF_HUB_CACHE`, `HF_DATASETS_CACHE`); only the token-cache and default
output paths are hardcoded in the code.
