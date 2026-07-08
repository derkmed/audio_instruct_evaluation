# eval — evaluation harness

Batched evaluation of audio-instruction models (Gemma, Qwen3-Omni) on the
Universal Audio Understanding dataset. Rows come from the shared
[`uad_data`](../uad_data) loader — the same rows [`train/`](../train) finetunes on.

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # private dataset; some models are gated
python -m eval.main --model GEMMA-4 --json-config configs/clotho_config.json --split test
```

Or run on an A100 in Colab: [`colab_eval.ipynb`](./colab_eval.ipynb).

## Flow

```mermaid
flowchart TD
    CLI["eval.main<br/>(CLI flags → EvalConfig)"] --> LOADER["uad_data.load_uad_dataset<br/>(audio + metadata + prompts from HF Hub)"]
    CLI --> BE["ModelBackend<br/>GemmaBackend / QwenBackend"]
    LOADER -- "row dicts" --> EV["Evaluator.evaluate"]

    subgraph BATCH["per batch (preprocessing overlaps GPU inference)"]
        PRE["preprocess_audio (uad_data.audio_utils)<br/>decode → 16 kHz mono float32, thread pool"]
        REQ["InferenceRequest batch<br/>(audio + sys_inst + prompt + ground truth)"]
        GEN["backend.generate_batch<br/>(one forward pass; falls back to sequential)"]
        PRE --> REQ --> GEN
    end

    EV --> BATCH
    GEN -- "predictions" --> JSONL["results.jsonl<br/>(flushed per sample — crash-safe)"]
    GEN -- "predictions + references" --> WER["WER (hf evaluate)"]
    WER --> SUM["summary.json + console report"]
```

## Pieces

| file | role |
| --- | --- |
| `main.py` | CLI entry point; wires config → loader → backend → evaluator |
| `config.py` | `EvalConfig` + `DEFAULT_MODEL_PATHS` (registry shared with `train/`) |
| `evaluator.py` | batch loop: threaded audio preprocessing overlapping GPU inference, incremental `results.jsonl`, WER |
| `backends/base.py` | `ModelBackend` ABC + `InferenceRequest` |
| `backends/gemma.py` | Gemma: audio arrays in chat messages, batched `processor(text, audio)` |
| `backends/qwen.py` | Qwen3-Omni: temp WAVs + `process_mm_info`, batched processing |

To evaluate a finetuned checkpoint, merge the LoRA adapter and pass it via
`--model-path` — see [`FINETUNING.md`](../FINETUNING.md#evaluating-a-finetuned-model).
