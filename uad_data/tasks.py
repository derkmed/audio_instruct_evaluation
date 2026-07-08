"""Audio-understanding task definitions.

`Task` enumerates every task the dataset supports and, via `Task.features`, the
metadata column(s) each task expects. Those column names are what
`sample.Sample` pulls from a record to render the prompt / instruction / output
templates, so the keys here must match the fields present in the per-split
metadata JSONs and the placeholders used in `prompts/*.json`.
"""
import datasets
import enum
from typing import Any


class Task(enum.Enum):
    """Enum class for Audio Understanding tasks."""
    ASR = "asr"
    ASR_TIMESTAMP_SEARCH = "asr_timestamp_search"
    CLASSIFICATION = "classification"
    # Classification tasks should be suffixed with 'classification'.
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CAPTION = "caption"
    COMMONSENSE = "commonsense"
    COMMONSENSE_HARD = "commonsense_hard"
    QA = "qa"
    ENGLISH_TRANSLATION = "english_translation"
    INTONATION_DETECTION = "intonation_detection"
    INTENT_DETECTION = "intent_detection"
    INTENT_DETECTION_NL = "intent_detection_nl"
    ACTION_CLASSIFICATION = "action_classification"

    @property
    def features(self) -> dict[str, Any]:
        """Maps each task to the corresponding feature column names.

        Each task has an expected feature output. This is linked here.
        """
        if self == Task.CLASSIFICATION:
            return {
                # Ground truth category for this sample.
                'category': datasets.Value('string'),
                # Comma-delimited string of all available categories.
                'categories': datasets.Value('string'),
            }
        elif self == Task.ASR_TIMESTAMP_SEARCH:
            return {
                'transcriptions': {
                    "start_time": datasets.Value("float"),
                    "end_time": datasets.Value("float"),
                    "transcription": datasets.Value("string")
                }
            }
        elif self == Task.ASR:
            return {'transcription': datasets.Value('string')}
        elif self == Task.SENTIMENT_ANALYSIS:
            return {'Sentiment': datasets.Value('string')}
        elif self == Task.CAPTION:
            return {'caption': datasets.Value('string')}
        elif self == Task.COMMONSENSE:
            return {
                'commonsense_choices': datasets.Value('string'),
                'commonsense_answer': datasets.Value('string')
            }
        elif self == Task.COMMONSENSE_HARD:
            return {
                'hard_commonsense_choices': datasets.Value('string'),
                'hard_commonsense_answer': datasets.Value('string')
            }
        elif self == Task.QA:
            return {'question': datasets.Value('string'), 'answer': datasets.Value('string')}
        elif self == Task.ENGLISH_TRANSLATION:
            return {"english_translation": datasets.Value("string")}
        elif self == Task.INTENT_DETECTION_NL:
            return {"intent_nl": datasets.Value("string")}
        elif self == Task.INTENT_DETECTION:
            return {"intent": datasets.Value("string")}
        elif self == Task.ACTION_CLASSIFICATION:
            return {"action": datasets.Value("string")}
        elif self == Task.INTONATION_DETECTION:
            return {"category": datasets.Value("string")}
        else:
            raise NotImplementedError(
                f'{self.value} prompt handling not yet implemented.')

    def __lt__(self, other):
        return self.value < other.value
