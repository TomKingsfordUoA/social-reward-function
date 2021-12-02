import dataclasses
import typing

import dataclasses_json
import pandas as pd  # type: ignore


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class EmotionWeights:
    overall: float
    angry: float
    disgusted: float
    fearful: float
    happy: float
    sad: float
    surprised: float
    neutral: float


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class PresenceWeights:
    accompanied: float
    alone: float


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class RewardSignalConfig:
    audio_weights: EmotionWeights
    video_weights: EmotionWeights
    presence_weights: PresenceWeights
    period_s: float
    threshold_audio_power: float
    threshold_latency_s: float

    @property
    def s_video_coefficients(self) -> pd.Series:
        return pd.Series({
            'angry': self.video_weights.angry,
            'disgusted': self.video_weights.disgusted,
            'fearful': self.video_weights.fearful,
            'happy': self.video_weights.happy,
            'sad': self.video_weights.sad,
            'surprised': self.video_weights.surprised,
            'neutral': self.video_weights.neutral,
        })

    @property
    def s_audio_coefficients(self) -> pd.Series:
        return pd.Series({
            'angry': self.audio_weights.angry,
            'disgusted': self.audio_weights.disgusted,
            'fearful': self.audio_weights.fearful,
            'happy': self.audio_weights.happy,
            'sad': self.audio_weights.sad,
            'surprised': self.audio_weights.surprised,
            'neutral': self.audio_weights.neutral,
        })


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class AudioInputConfig:
    period_propn: float
    segment_duration_s: float


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class FileOutputConfig:
    enabled: bool
    path: str
    overwrite: bool


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass
class VisualizationOutputConfig:
    reward_window_width_s: float
    threshold_lag_s: float
    moving_average_window_width_s: float
    display_video: bool
    display_plots: bool


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class VideoInputConfig:
    target_fps: float


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class FileInputConfig:
    path: str
    play_audio: bool


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class WebcamInputConfig:
    device_id: int


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class DatasetInputConfig:
    path: str


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class InputSourceConfig:
    file: typing.Optional[FileInputConfig] = dataclasses.field(default=None)
    webcam: typing.Optional[WebcamInputConfig] = dataclasses.field(default=None)
    dataset: typing.Optional[DatasetInputConfig] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        if not ((self.file is None) ^ (self.webcam is None) ^ (self.dataset is None)):
            raise ValueError("Exactly one of 'webcam', 'file' and 'dataset' must be specified")


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class InputConfig:
    audio: AudioInputConfig
    video: VideoInputConfig
    source: InputSourceConfig


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class OutputConfig:
    visualization: VisualizationOutputConfig
    file: FileOutputConfig


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class Config:
    input: InputConfig
    reward_signal: RewardSignalConfig
    output: OutputConfig
