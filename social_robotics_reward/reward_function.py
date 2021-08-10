import asyncio
import ctypes
import dataclasses
import multiprocessing
import queue
import time
from math import floor
from typing import Optional, cast, AsyncGenerator, Generator, Dict, Tuple, List

import pandas as pd  # type: ignore

from emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer
from emotion_recognition_using_speech.utils import get_best_estimators
from residual_masking_network import RMN
from social_robotics_reward.sensors.audio import AudioFrame
from social_robotics_reward.sensors.video import VideoFrame
from social_robotics_reward.util import CodeBlockTimer


Timestamp = float


@dataclasses.dataclass(frozen=True)
class RewardSignal:
    timestamp_s: float
    combined_reward: float
    audio_reward: Optional[float]
    video_reward: Optional[float]
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


class RewardFunction:
    def __init__(
            self,
            period_s: float,
    ) -> None:

        self._period_s = period_s

        self._queue_video_frames: queue.Queue[VideoFrame] = multiprocessing.Queue()
        self._queue_audio_frames: queue.Queue[AudioFrame] = multiprocessing.Queue()

        self._semaphore_reward_signal = multiprocessing.Semaphore(value=0)
        self._queue_reward_signal: queue.Queue[RewardSignal] = multiprocessing.Queue()

        self._proc = multiprocessing.Process(target=self._gen)
        self._is_running = multiprocessing.Value(ctypes.c_bool)
        self._is_running.value = True

    @staticmethod
    def _load_audio_classifier() -> EmotionRecognizer:
        estimators = get_best_estimators(classification=True)  # type: ignore
        best_estimator = max(estimators, key=lambda elem: cast(float, elem[2]))  # elem[2] is accuracy
        return EmotionRecognizer(best_estimator[0])  # type: ignore

    @staticmethod
    def _load_video_classifier() -> RMN:
        return RMN()  # type: ignore

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

    def gen(self) -> Generator[RewardSignal, None, None]:
        if self._proc.is_alive():
            raise RuntimeError(f"{RewardFunction.__name__} already running")
        self._proc.start()
        while self._proc.is_alive() or not self._queue_audio_frames.empty() or not self._queue_video_frames.empty():
            if not self._semaphore_reward_signal.acquire(timeout=1e-1):
                continue
            elem = self._queue_reward_signal.get()
            yield elem

    async def gen_async(self) -> AsyncGenerator[RewardSignal, None]:
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
        _video_classifier = RewardFunction._load_video_classifier()
        _audio_classifier = RewardFunction._load_audio_classifier()

        buffer_video_frames = [self._queue_video_frames.get(block=True, timeout=None)]
        emotions_video_frames: List[Tuple[Timestamp, Dict[str, float]]] = []
        buffer_audio_frames = [self._queue_audio_frames.get(block=True, timeout=None)]
        emotions_audio_frames: List[Tuple[Timestamp, Dict[str, float]]] = []
        timestamp_last_reward_signal = min(buffer_video_frames[0].timestamp_s, buffer_audio_frames[0].timestamp_s)
        wallclock_initial = time.time()
        idx_reward_signal = 0

        while (self._is_running.value or
               not self._queue_video_frames.empty() or
               not self._queue_audio_frames.empty() or
               len(buffer_video_frames) != 0 or
               len(emotions_video_frames) != 0 or
               len(buffer_audio_frames) != 0 or
               len(emotions_audio_frames) != 0):

            expected_idx_reward_signal = floor((time.time() - wallclock_initial) / self._period_s)

            # Do we need to skip a reward period?
            if expected_idx_reward_signal - idx_reward_signal > 1:
                print("Warning! Reward signal falling behind - skipping release")
                idx_reward_signal = expected_idx_reward_signal - 1

            # Are we ready to release a reward signal?
            if expected_idx_reward_signal - idx_reward_signal == 1:
                idx_reward_signal += 1

                included_emotions_video_frames = [
                    predicted_emotions for timestamp_s, predicted_emotions in emotions_video_frames
                    if timestamp_s <= timestamp_last_reward_signal + self._period_s
                ]
                emotions_video_frames = [
                    (timestamp_s, predicted_emotions) for timestamp_s, predicted_emotions in emotions_video_frames
                    if timestamp_s > timestamp_last_reward_signal + self._period_s
                ]
                included_emotions_audio_frames = [
                    predicted_emotions for timestamp_s, predicted_emotions in emotions_audio_frames
                    if timestamp_s <= timestamp_last_reward_signal + self._period_s
                ]
                emotions_audio_frames = [
                    (timestamp_s, predicted_emotions) for timestamp_s, predicted_emotions in emotions_audio_frames
                    if timestamp_s > timestamp_last_reward_signal + self._period_s
                ]

                df_video_emotions = pd.DataFrame(included_emotions_video_frames)
                df_audio_emotions = pd.DataFrame(included_emotions_audio_frames)

                # Calculate the combined reward:
                wt_audio = 1.0
                wt_video = 1.0
                s_audio_coefficients = pd.Series({
                    'happy': 1.0,
                    'neutral': 0.0,
                    'sad': -1.0,
                })
                s_video_coefficients = pd.Series({
                    'angry': -1.0,
                    'disgust': -1.0,
                    'fear': -1.0,
                    'happy': 1.0,
                    'sad': -1.0,
                    'surprise': 0.0,
                    'neutral': 0.0,
                })
                if not df_audio_emotions.empty and set(s_audio_coefficients.index) != set(df_audio_emotions.columns):
                    raise ValueError(f"Unexpected audio emotions: got {df_audio_emotions.columns} expected {s_audio_coefficients.index}")
                if not df_video_emotions.empty and set(s_video_coefficients.index) != set(df_video_emotions.columns):
                    raise ValueError(f"Unexpected video emotions: got {df_video_emotions.columns} expected {s_video_coefficients.index}")
                audio_reward = None if df_audio_emotions.empty else (s_audio_coefficients * df_audio_emotions).to_numpy().sum()
                video_reward = None if df_video_emotions.empty else (s_video_coefficients * df_video_emotions).to_numpy().sum()
                combined_reward = (
                        wt_audio * (audio_reward if audio_reward is not None else 0.0) +
                        wt_video * (video_reward if video_reward is not None else 0.0)
                )

                timestamp_last_reward_signal += self._period_s

                self._queue_reward_signal.put(RewardSignal(
                    timestamp_s=timestamp_last_reward_signal,
                    combined_reward=combined_reward,
                    video_reward=video_reward,
                    audio_reward=audio_reward,
                    detected_video_emotions=df_video_emotions,
                    detected_audio_emotions=df_audio_emotions,
                ))
                self._semaphore_reward_signal.release()

            # Clear the queues:
            while not self._queue_audio_frames.empty():
                buffer_audio_frames.append(self._queue_audio_frames.get_nowait())
            while not self._queue_video_frames.empty():
                buffer_video_frames.append(self._queue_video_frames.get_nowait())

            # Emotion predictions:
            with CodeBlockTimer() as timer:
                emotions_video_frames += [
                    (frame.timestamp_s, {key: value for emotion in face['proba_list'] for key, value in emotion.items()})
                    for frame in buffer_video_frames
                    for face in _video_classifier.detect_emotion_for_single_frame(frame=frame.video_data)]
            if len(buffer_video_frames) != 0:
                print(f'Video prediction took {timer.timedelta} '
                      f'(={timer.timedelta / len(buffer_video_frames) if len(buffer_video_frames) else "NaN"} per frame)')
            buffer_video_frames.clear()
            with CodeBlockTimer() as timer:
                emotions_audio_frames += [
                    (frame.timestamp_s, _audio_classifier.predict_proba(audio_data=frame.audio_data, sample_rate=frame.sample_rate))
                    for frame in buffer_audio_frames]
            if len(buffer_audio_frames) != 0:
                print(f'Audio prediction took {timer.timedelta}')
            buffer_audio_frames.clear()
