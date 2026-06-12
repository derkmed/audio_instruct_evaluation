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
