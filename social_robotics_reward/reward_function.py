import asyncio
import ctypes
import dataclasses
import math
import multiprocessing
import queue
import sys
import time
import typing

import dataclasses_json
import numpy as np
import pandas as pd  # type: ignore

import emotion_recognition_using_speech.emotion_recognition
from mevonai_speech_emotion_recognition.src.speechEmotionRecognition import \
    EmotionRecognizer as MevonAiEmotionRecognizer
from residual_masking_network.rmn import RMN
from social_robotics_reward.sensors.audio import AudioFrame
from social_robotics_reward.sensors.video import VideoFrame
from social_robotics_reward.util import CodeBlockTimer

Timestamp = float


@dataclasses.dataclass(frozen=True)
class EmotionProbabilities:
    happy: typing.Optional[float] = dataclasses.field(default=None)
    neutral: typing.Optional[float] = dataclasses.field(default=None)
    sad: typing.Optional[float] = dataclasses.field(default=None)
    angry: typing.Optional[float] = dataclasses.field(default=None)
    fearful: typing.Optional[float] = dataclasses.field(default=None)
    disgusted: typing.Optional[float] = dataclasses.field(default=None)
    surprised: typing.Optional[float] = dataclasses.field(default=None)


class VideoEmotionRecognizer:
    def detect_emotion_for_single_frame(self, frame: typing.Any) -> typing.Iterable[EmotionProbabilities]:
        raise NotImplementedError()


class RMNVideoEmotionRecognizer(VideoEmotionRecognizer):
    _emotions = frozenset(('neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'))

    def __init__(self) -> None:
        self._rmn = RMN()  # type: ignore

    def detect_emotion_for_single_frame(self, frame: typing.Any) -> typing.Iterable[EmotionProbabilities]:
        emotion_probabilities_all_faces = [
            {
                key: value
                for emotion in face['proba_list']
                for key, value in emotion.items()
             }
            for face in self._rmn.detect_emotion_for_single_frame(frame=frame)
        ]
        for emotion_probabilities in emotion_probabilities_all_faces:
            if len(set(emotion_probabilities.keys()) - self._emotions) != 0:
                raise ValueError("Model returned unexpected emotions")
        return [
            EmotionProbabilities(
                happy=emotion_probabilities['happy'],
                neutral=emotion_probabilities['neutral'],
                sad=emotion_probabilities['sad'],
                angry=emotion_probabilities['angry'],
                fearful=emotion_probabilities['fear'],
                disgusted=emotion_probabilities['disgust'],
                surprised=emotion_probabilities['surprise'],
            )
            for emotion_probabilities in emotion_probabilities_all_faces
        ]


class AudioEmotionRecognizer:
    def predict_proba(self, audio_data: typing.Any, sample_rate: int) -> EmotionProbabilities:
        raise NotImplementedError()


class MevonAIAudioEmotionRecognizer(AudioEmotionRecognizer):
    _emotions = frozenset(('Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised'))

    def __init__(self, model_file: typing.Optional[str] = None) -> None:
        self._emotion_recognizer = MevonAiEmotionRecognizer(model_file=model_file)

    def predict_proba(self, audio_data: typing.Any, sample_rate: int) -> EmotionProbabilities:
        emotion_probabilities = self._emotion_recognizer.predict_proba(audio_data=audio_data, sample_rate=sample_rate)

        if len(set(emotion_probabilities.keys()) - self._emotions) != 0:
            raise ValueError("Model returned unexpected emotions")

        for emotion in emotion_probabilities:
            if emotion_probabilities[emotion] is None:
                emotion_probabilities[emotion] = 0.0

        return EmotionProbabilities(
            happy=emotion_probabilities['Happy'],
            neutral=emotion_probabilities['Neutral'],
            sad=emotion_probabilities['Sad'],
            angry=emotion_probabilities['Angry'],
            fearful=emotion_probabilities['Fearful'],
            disgusted=emotion_probabilities['Disgusted'],
            surprised=emotion_probabilities['Surprised'],
        )


class ERUSAudioEmotionRecognizer(AudioEmotionRecognizer):
    _emotions = frozenset(('neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom'))

    def __init__(self) -> None:
        estimators = emotion_recognition_using_speech.emotion_recognition.get_best_estimators(classification=True)  # type: ignore
        best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is accuracy
        self._emotion_recognizer = emotion_recognition_using_speech.emotion_recognition.EmotionRecognizer(  # type: ignore
            model=best_estimator[0],
            emotions=["happy", "neutral", "sad"],
            balance=True)

    def predict_proba(self, audio_data: typing.Any, sample_rate: int) -> EmotionProbabilities:
        emotion_probabilities = self._emotion_recognizer.predict_proba(audio_data=audio_data, sample_rate=sample_rate)

        if len(set(emotion_probabilities.keys()) - self._emotions) != 0:
            raise ValueError("Model returned unexpected emotions")

        for emotion in emotion_probabilities:
            if emotion_probabilities[emotion] is None:
                emotion_probabilities[emotion] = 0.0

        return EmotionProbabilities(
            happy=emotion_probabilities['happy'],
            neutral=emotion_probabilities['neutral'],
            sad=emotion_probabilities['sad'],
        )


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
class RewardSignalConfig:
    audio_weights: EmotionWeights
    video_weights: EmotionWeights
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


@dataclasses.dataclass(frozen=True)
class RewardSignal:
    timestamp_s: float
    combined_reward: float
    audio_reward: typing.Optional[float]
    video_reward: typing.Optional[float]
    detected_audio_emotions: pd.DataFrame
    detected_video_emotions: pd.DataFrame

    def __repr__(self) -> str:
        return f"RewardSignal(\n" \
               f"\ttimestamp_s={self.timestamp_s}\n" \
               f"\tcombined_reward={self.combined_reward}\n" \
               f"\taudio_reward={self.audio_reward}\n"\
               f"\tvideo_reward={self.video_reward}\n" \
               f"\tdetected_audio_emotions=\n{self.detected_audio_emotions.mean()}\n" \
               f"\tdetected_video_emotions=\n{self.detected_video_emotions.mean()}\n" \
               f")"

    def __add__(self, other: 'RewardSignal') -> 'RewardSignal':
        audio_reward = 0.0
        if self.audio_reward is not None:
            audio_reward += self.audio_reward
        if other.audio_reward is not None:
            audio_reward += other.audio_reward

        video_reward = 0.0
        if self.video_reward is not None:
            video_reward += self.video_reward
        if other.video_reward is not None:
            video_reward += other.video_reward

        return RewardSignal(
            timestamp_s=max(self.timestamp_s, other.timestamp_s),
            combined_reward=self.combined_reward + other.combined_reward,
            audio_reward=audio_reward,
            video_reward=video_reward,
            detected_audio_emotions=pd.concat([self.detected_audio_emotions, other.detected_audio_emotions]),
            detected_video_emotions=pd.concat([self.detected_video_emotions, other.detected_video_emotions]),
        )

    def __iadd__(self, other: 'RewardSignal') -> 'RewardSignal':
        return self + other

    def __truediv__(self, other: typing.Union[int, float]) -> 'RewardSignal':
        return RewardSignal(
            timestamp_s=self.timestamp_s,
            combined_reward=self.combined_reward / other,
            audio_reward=self.audio_reward / other if self.audio_reward is not None else None,
            video_reward=self.video_reward / other if self.video_reward is not None else None,
            detected_audio_emotions=self.detected_audio_emotions,
            detected_video_emotions=self.detected_video_emotions,
        )

    def __itruediv__(self, other: typing.Union[int, float]) -> 'RewardSignal':
        return self / other


class RewardFunction:
    def __init__(self, config: RewardSignalConfig) -> None:
        self._config = config

        self._queue_video_frames: queue.Queue[VideoFrame] = multiprocessing.Queue()
        self._queue_audio_frames: queue.Queue[AudioFrame] = multiprocessing.Queue()

        self._semaphore_reward_signal = multiprocessing.Semaphore(value=0)
        self._queue_reward_signal: queue.Queue[RewardSignal] = multiprocessing.Queue()

        self._proc = multiprocessing.Process(target=self._gen)
        self._is_running = multiprocessing.Value(ctypes.c_bool)
        self._is_running.value = True

    def __enter__(self) -> 'RewardFunction':
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        pass

    @staticmethod
    def _load_audio_classifiers() -> typing.Iterable[AudioEmotionRecognizer]:
        return ERUSAudioEmotionRecognizer(), MevonAIAudioEmotionRecognizer(),

    @staticmethod
    def _load_video_classifiers() -> typing.Iterable[VideoEmotionRecognizer]:
        return RMNVideoEmotionRecognizer(),

    def stop(self) -> None:
        print("stopped")
        self._is_running.value = False

    async def stop_async(self, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        print("stopped")
        self._is_running.value = False

    def push_video_frame(self, video_frame: VideoFrame) -> None:
        print(f"{RewardFunction.__name__} got video frame t={video_frame.timestamp_s}", flush=True)
        self._queue_video_frames.put(video_frame)

    def push_audio_frame(self, audio_frame: AudioFrame) -> None:
        print(f"{RewardFunction.__name__} got audio frame t={audio_frame.timestamp_s}", flush=True)
        self._queue_audio_frames.put(audio_frame)

    def gen(self) -> typing.Generator[RewardSignal, None, None]:
        if self._proc.is_alive():
            raise RuntimeError(f"{RewardFunction.__name__} already running")
        self._proc.start()
        while self._proc.is_alive() or not self._queue_audio_frames.empty() or not self._queue_video_frames.empty():
            if not self._semaphore_reward_signal.acquire(timeout=1e-1):
                continue
            elem = self._queue_reward_signal.get()
            yield elem

    async def gen_async(self) -> typing.AsyncGenerator[RewardSignal, None]:
        try:
            if self._proc.is_alive():
                raise RuntimeError(f"{RewardFunction.__name__} already running")
            self._proc.start()
            while self._proc.is_alive() or not self._queue_reward_signal.empty():
                if not self._semaphore_reward_signal.acquire(block=False):
                    await asyncio.sleep(0)
                    continue
                elem = self._queue_reward_signal.get()
                yield elem
        except asyncio.CancelledError:
            pass

    def _gen(self) -> None:
        # Initialize emotion classifiers:
        _video_classifiers = RewardFunction._load_video_classifiers()
        _audio_classifiers = RewardFunction._load_audio_classifiers()

        buffer_video_frames = [self._queue_video_frames.get(block=True, timeout=None)]
        emotions_video_frames: typing.List[typing.Tuple[Timestamp, typing.Iterable[EmotionProbabilities]]] = []
        buffer_audio_frames = [self._queue_audio_frames.get(block=True, timeout=None)]
        emotions_audio_frames: typing.List[typing.Tuple[Timestamp, EmotionProbabilities]] = []
        wallclock_initial = time.time()
        wallclock_next = wallclock_initial + self._config.period_s

        while (self._is_running.value or
               not self._queue_video_frames.empty() or
               not self._queue_audio_frames.empty() or
               len(buffer_video_frames) != 0 or
               len(emotions_video_frames) != 0 or
               len(buffer_audio_frames) != 0 or
               len(emotions_audio_frames) != 0):

            now = time.time()

            # Do we need to skip release period(s)?
            lag = now - wallclock_next
            skip_release_periods = lag > self._config.threshold_latency_s
            if skip_release_periods:
                wallclock_pre_correction = wallclock_next
                wallclock_next += math.ceil(lag / self._config.period_s) * self._config.period_s
                print(f"Warning! Reward signal fell behind. lag={lag:.2f}. Skipping release(s) to catch up "
                      f"[{wallclock_pre_correction - wallclock_initial:.2f}, {wallclock_next - wallclock_initial})",
                      file=sys.stderr)
                lag = now - wallclock_next
                assert lag <= 0, lag  # sanity check - negative lag after correction
            assert lag <= self._config.threshold_latency_s, lag  # sanity check - lag never more than threshold

            # Are we ready to release a reward signal?
            if lag >= 0:
                timestamp_next = wallclock_next - wallclock_initial
                timestamp_prev = timestamp_next - self._config.period_s

                # Clean up skipped data:
                # If we aren't lagging exceeding threshold, we'd rather occasionally include data from a previous period
                # than lose the data
                # FIXME(TK): sanity check there is no data which is *too* old being included!
                if skip_release_periods:
                    emotions_video_frames = [(timestamp_s, predicted_emotions)
                                             for timestamp_s, predicted_emotions in emotions_video_frames
                                             if timestamp_s > timestamp_prev]
                    emotions_audio_frames = [(timestamp_s, predicted_emotions)
                                             for timestamp_s, predicted_emotions in emotions_audio_frames
                                             if timestamp_s > timestamp_prev]

                included_emotions_video_frames = [
                    predicted_emotions for timestamp_s, predicted_emotions in emotions_video_frames
                    if timestamp_s <= timestamp_next
                ]
                emotions_video_frames = [
                    (timestamp_s, predicted_emotions) for timestamp_s, predicted_emotions in emotions_video_frames
                    if timestamp_s > timestamp_next
                ]
                included_emotions_audio_frames = [
                    predicted_emotions for timestamp_s, predicted_emotions in emotions_audio_frames
                    if timestamp_s <= timestamp_next
                ]
                emotions_audio_frames = [
                    (timestamp_s, predicted_emotions) for timestamp_s, predicted_emotions in emotions_audio_frames
                    if timestamp_s > timestamp_next
                ]

                df_video_emotions = pd.DataFrame([
                    emotion_probs.__dict__
                    for emotion_probs_all_faces in included_emotions_video_frames
                    for emotion_probs in emotion_probs_all_faces
                ])
                df_audio_emotions = pd.DataFrame([
                    emotion_probs.__dict__
                    for emotion_probs in included_emotions_audio_frames
                ])

                # NaN means no value provided by estimator, so replace with 0.0:
                df_video_emotions.fillna(0.0, inplace=True)
                df_audio_emotions.fillna(0.0, inplace=True)

                # Calculate the combined reward:
                if not df_audio_emotions.empty and set(self._config.s_audio_coefficients.index) != set(df_audio_emotions.columns):
                    raise ValueError(f"Unexpected audio emotions: got {df_audio_emotions.columns} expected {self._config.s_audio_coefficients.index}")
                if not df_video_emotions.empty and set(self._config.s_video_coefficients.index) != set(df_video_emotions.columns):
                    raise ValueError(f"Unexpected video emotions: got {df_video_emotions.columns} expected {self._config.s_video_coefficients.index}")
                audio_reward = 0.0 if df_audio_emotions.empty else (self._config.s_audio_coefficients * df_audio_emotions).to_numpy().sum()
                video_reward = 0.0 if df_video_emotions.empty else (self._config.s_video_coefficients * df_video_emotions).to_numpy().sum()
                combined_reward = self._config.audio_weights.overall * audio_reward + self._config.video_weights.overall * video_reward

                if not math.isfinite(video_reward):
                    raise ValueError(f'video_reward is not finite! {video_reward}')
                if not math.isfinite(audio_reward):
                    raise ValueError(f'audio_reward is not finite! {audio_reward}')
                if not math.isfinite(combined_reward):
                    raise ValueError(f'combined_reward is not finite! {combined_reward}')

                self._queue_reward_signal.put(RewardSignal(
                    timestamp_s=timestamp_next,
                    combined_reward=combined_reward,
                    video_reward=video_reward,
                    audio_reward=audio_reward,
                    detected_video_emotions=df_video_emotions,
                    detected_audio_emotions=df_audio_emotions,
                ))
                self._semaphore_reward_signal.release()

                wallclock_next += self._config.period_s

            # Clear the queues:
            while not self._queue_audio_frames.empty():
                buffer_audio_frames.append(self._queue_audio_frames.get_nowait())
            while not self._queue_video_frames.empty():
                buffer_video_frames.append(self._queue_video_frames.get_nowait())

            # Emotion predictions:
            with CodeBlockTimer() as timer:
                for video_classifier in _video_classifiers:
                    emotions_video_frames += [
                        (frame.timestamp_s, video_classifier.detect_emotion_for_single_frame(frame.video_data))
                        for frame in buffer_video_frames
                    ]
            if len(buffer_video_frames) != 0:
                print(f'Video prediction took {timer.timedelta} '
                      f'(={timer.timedelta / len(buffer_video_frames) if len(buffer_video_frames) else "NaN"} per frame)')
            buffer_video_frames.clear()

            buffer_audio_frames = [frame for frame in buffer_audio_frames if np.mean(np.power(frame.audio_data, 2)) >= self._config.threshold_audio_power]
            with CodeBlockTimer() as timer:
                for audio_classifier in _audio_classifiers:
                    emotions_audio_frames += [
                        (frame.timestamp_s, audio_classifier.predict_proba(audio_data=frame.audio_data, sample_rate=frame.sample_rate))
                        for frame in buffer_audio_frames
                    ]
            if len(buffer_audio_frames) != 0:
                print(f'Audio prediction took {timer.timedelta}')
            buffer_audio_frames.clear()
