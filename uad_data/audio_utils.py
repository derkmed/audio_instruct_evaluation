"""Audio preprocessing shared by the eval and train model backends.

Turns the raw audio bytes carried on each dataset row into the normalized
float32 mono array the model processors expect. Lives in `uad_data` (rather than
`eval/` or `train/`) because both harnesses need identical preprocessing so that
finetuned models are trained and evaluated on the same input distribution.
"""
import io

import librosa
import numpy as np
import soundfile as sf


def preprocess_audio(
    audio_bytes: bytes,
    target_sr: int = 16_000,
    max_seconds: int = 30,
) -> np.ndarray:
    """Decode bytes → float32 mono array at target_sr, capped at max_seconds."""
    buf = io.BytesIO(audio_bytes)
    arr, orig_sr = sf.read(buf, dtype="float32")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if orig_sr != target_sr:
        arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr, res_type="scipy")

    max_samples = max_seconds * target_sr
    return arr[:max_samples]
