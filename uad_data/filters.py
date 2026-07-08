"""Per-sample inclusion filters applied during row generation.

A `SampleFilter.include_sample(sample)` decides whether an expanded row is kept.
Filters are selected by name in a run config's optional `sample_filter` field
(`from_config`); the default is `AllPassFilter` (keep everything).
"""

import abc
import random

from . import sample

Sample = sample.Sample

class SampleFilter(abc.ABC):
    """Use this filter to determine whether a sample should be included in a dataset or not."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def include_sample(self, sample: Sample) -> bool:
        raise NotImplementedError

class AllPassFilter(SampleFilter):
    """All samples are filtered in."""

    def include_sample(self, sample: Sample) -> bool:
        return True


class RandomFilter(SampleFilter):
    """All samples are selected randomly."""

    def __init__(self, random_seed: int = 42):
        super().__init__()
        random.seed(random_seed)

    def include_sample(self, sample: Sample) -> bool:
        if random.random() > 0.5:
            return True
        else:
            return False


FILTER_REGISTRY: dict[str, type[SampleFilter]] = {
    'all_pass': AllPassFilter,
    'random': RandomFilter,
}


def from_config(name: str, **kwargs) -> SampleFilter:
    if name not in FILTER_REGISTRY:
        raise ValueError(
            f'Unknown sample_filter "{name}". Must be one of: {list(FILTER_REGISTRY)}')
    return FILTER_REGISTRY[name](**kwargs)
