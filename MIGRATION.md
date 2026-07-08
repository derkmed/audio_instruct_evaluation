# Migrating off the HuggingFace loading script

This repo now contains **both** the evaluation harness and the dataset-generation
library that used to live inside the HF dataset repo as a `trust_remote_code`
loading script. The audio data itself stays on the (private) HF Hub.

## What changed

- New package **`uad_data/`** — the dataset-generation library, moved out of the
  HF repo and turned into normal importable Python:
  - `tasks.py`, `io_templates.py`, `prompts.py`, `sample.py`, `filters.py`,
    `internal_dataset.py`, `internal_datasets.py`, `json_config_loader.py`
    (moved verbatim, only imports made package-relative);
  - `collection.py` — plain replacement for the old `UniversalAudioUnderstandingConfig`
    builder-config (no `datasets.GeneratorBasedBuilder`);
  - `hub.py` — downloads audio archives, metadata, prompts and configs from the
    Hub with `huggingface_hub` (replaces `dl_manager` + the `_ensure_hub_resources`
    runtime shim);
  - `loader.py` — `load_uad_dataset(...)`, the drop-in replacement for
    `load_dataset(..., trust_remote_code=True)`. Streams each `tar.gz`, expands
    every `(audio × task × prompt-template)` combination, and returns row dicts
    with the **same schema** the loading script produced.
- The eval harness moved into an **`eval/`** package (`eval/main.py`, `config.py`,
  `evaluator.py`, `audio_utils.py`, `backends/`), with intra-package imports made
  relative. Run it from the repo root with `python -m eval.main ...`.
- `eval/main.py` and `colab_eval.ipynb` now call `load_uad_dataset(...)` instead of
  `load_dataset(..., trust_remote_code=True)`.
- `requirements.txt` gains `huggingface_hub` and `jinja2` as direct deps.
- `tests/test_loader.py` — offline end-to-end test (synthetic archive, no network).

Two bugs were fixed in passing:
- **Windows path separators**: `split_metadata_path` used `os.path.join`, which
  emits `data\...\...` on Windows and breaks Hub paths; it now uses `/`
  explicitly, and `hub.to_repo_path` normalizes any stray backslashes.
- **Row aliasing**: `Sample.to_output()` reused the shared metadata dict, so
  materializing the generator into a list aliased every expansion of one audio
  clip to the last one. It now returns an independent shallow copy. (Invisible
  under the old script because each yield was serialized straight to Arrow.)

## What stays on the HF repo (do NOT delete)

These are plain data files the loader downloads at runtime — keep them:

- `data/**` — the audio archives and per-split metadata JSONs.
- `prompts/*.json` — your system-instruction / prompt / output templates.
- `universal_audio_dataset_configs/*.json` — named run configs.
- `README.md`, `.gitattributes`.

## HF-repo cleanup (prepare-don't-delete — run only when confident)

Once you've validated a real run against the Hub, these files in the **HF dataset
repo** are now duplicated by `uad_data/` and can be removed there so the code has
a single source of truth in this repo:

```
Universal-Audio-Understanding.py   # the loading script itself
tasks.py
io_templates.py
prompts.py
sample.py
filters.py
internal_dataset.py
internal_datasets.py
json_config_loader.py
__init__.py                        # only if it exists solely to support the above
```

> Note: the HF repo also holds other Python tooling not touched by this migration
> (`gemini_inference.py`, `gemini_prepare_data.py`, `token_counter.py`, `main.py`,
> and the `utils/`, `evaluation/`, `report_scripts/` dirs). Those are a separate
> decision — decide per-file whether they belong in this code repo too.

## How to run

```bash
pip install -r requirements.txt
# run from the repo root (the eval harness is now the `eval` package)
python -m eval.main --model GEMMA-4 --json-config clotho_config.json --split test
```

`--json-config` accepts either a local path (e.g. `clotho_config.json`) or the
name of a config hosted in the repo's `universal_audio_dataset_configs/` folder.

## How to test (no GPU, no 48 GB download)

```bash
python tests/test_loader.py
```
Only requires `datasets`, `jinja2`, `huggingface_hub`.
