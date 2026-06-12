from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL_PATHS: dict[str, str] = {
    "GEMMA-4": "google/gemma-4-e2b-it",
    "QWEN3-Omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


@dataclass
class EvalConfig:
    model_choice: str  # "GEMMA-4" | "QWEN3-Omni"

    # Dataset
    dataset_name: str = "AudioInstruct/Universal-Audio-Understanding"
    dataset_split: str = "test"
    json_config_path: str = "clotho_config.json"

    # Model
    model_path: Optional[str] = None  # overrides DEFAULT_MODEL_PATHS if set
    max_new_tokens: int = 256

    # Performance
    batch_size: int = 4
    num_preprocessing_workers: int = 4  # threads for parallel audio preprocessing

    # Evaluation
    max_samples: Optional[int] = None  # None = full dataset
    output_dir: Optional[str] = None   # directory for results.jsonl + summary.json

    # Auth
    hf_token: Optional[str] = None

    # Audio preprocessing
    target_sr: int = 16_000
    max_audio_seconds: int = 30

    @property
    def resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        if self.model_choice not in DEFAULT_MODEL_PATHS:
            raise ValueError(
                f"Unknown model_choice '{self.model_choice}'. "
                f"Either set model_path or use one of: {list(DEFAULT_MODEL_PATHS)}"
            )
        return DEFAULT_MODEL_PATHS[self.model_choice]
