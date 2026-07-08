"""Configuration of Internal Datasets contained by Universal Audio Understanding Dataset."""
import datasets
from typing import Dict, List

from .tasks import Task

# The following is an agreed-upon filepath convention for the tar.gz data of each of the
# internal datasets.
TAR_GZ_FILEPATH = "data/{name}/{name}.tar.gz"

REPO_URL = 'https://huggingface.co/datasets/AudioInstruct/Universal-Audio-Understanding/resolve/main'


class InternalDataset:

    def __init__(
            self,
            name: str,
            tasks: list[Task] | Task,
            splits: list[datasets.Split] | datasets.Split,
            description: str = '',
            version: datasets.Version | None = datasets.Version("1.0.0"),
            data_url: str | None = None,
            lfs_mode: bool = True
        ):
        super().__init__()
        self.name = name
        self.description = description
        self.version = version
        self.set_tasks(tasks)
        self.set_splits(splits)
        self.repo_url = REPO_URL
        self.lfs_mode = lfs_mode
        self.data_url = (
            data_url if data_url else
            TAR_GZ_FILEPATH.format(name=self.name)
        )
        if self.lfs_mode:
            # Prepend with the Repo URL if the file is a git-lfs file.
            self.data_url = self.get_git_lfs_path(self.data_url)

    def get_git_lfs_path(self, metadata_path: str) -> str:
        return f'{self.repo_url}/{metadata_path}'

    def split_metadata_path(self, split: datasets.Split) -> str:
        """Returns the expected metadata path for a split of the dataset."""
        if split not in self._splits:
            raise ValueError(f'Split {split} not found in configured splits: {self._splits}.')
        # Use forward slashes explicitly: this is a Hub repo path, not a local OS
        # path, so os.path.join would wrongly emit backslashes on Windows.
        return self.get_git_lfs_path(
            f"data/{self.name}/{self.name}_{split}.json"
        )

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    def set_tasks(self, tasks: list[Task] | Task) -> None:
        self._tasks = [tasks] if not isinstance(tasks, List) else tasks

    def get_splits(self) -> list[datasets.Split]:
        return self._splits

    def set_splits(self, splits: list[datasets.Split] | datasets.Split) -> None:
        self._splits = [splits] if not isinstance(splits, List) else splits

    @property
    def features(self) -> Dict[str, datasets.Value]:
        _features = {
            "originating_dataset": datasets.Value("string"),
            "task": datasets.Value("string"),
            "audio_path": datasets.Value("string"),
            "audio": datasets.Audio(sampling_rate=8_000, decode=False),
            "system_instruction": datasets.Value("string"),
            "prompt": datasets.Value("string"),
            "output": datasets.Value("string"),
            "split": datasets.Value("string"),
        }
        for task in self.tasks:
            _features.update(task.features)
        return _features

    def __repr__(self) -> str:
        repr_text = ''
        repr_text += f'name: "{self.name}"'
        repr_text += f', version: "{str(self.version)}"'
        repr_text += f', tasks: {self._tasks}'
        repr_text += f', splits: {self._splits}'
        repr_text += f', description: "{self.description}"'
        repr_text += f', data_url: "{self.data_url}"'
        return f'<{repr_text}>'
