import argparse
import asyncio
import dataclasses
import signal
import typing
from typing import Any, Generator, Dict

import pandas as pd  # type: ignore

from emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer
from emotion_recognition_using_speech.utils import get_best_estimators
from residual_masking_network import RMN
from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, AudioFileFrameGenerator, \
    MicrophoneFrameGenerator, AudioFrame
from social_robotics_reward.generator_coroutine_combiner import interleave_temporally, GeneratorMeta
from social_robotics_reward.video_frame_generation import VideoFileFrameGenerator, VideoFrameGenerator, \
    WebcamFrameGenerator, VideoFrame


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


class RewardFunction:
    def __init__(self) -> None:
        self._queue_video_frames = asyncio.Queue()
        self._queue_audio_frames = asyncio.Queue()

        # Initialize emotion classifiers:
        self._audio_classifier = RewardFunction._load_audio_classifier()
        self._video_classifier = RMN()  # type: ignore

    @staticmethod
    def _load_audio_classifier() -> EmotionRecognizer:
        estimators = get_best_estimators(classification=True)  # type: ignore
        best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is accuracy
        return EmotionRecognizer(best_estimator[0])  # type: ignore

    async def push_video_frame(self, video_frame: VideoFrame) -> None:
        await self._queue_video_frames.put(video_frame)

    async def push_audio_frame(self, audio_frame: AudioFrame) -> None:
        await self._queue_audio_frames.put(audio_frame)

    async def gen(self, period_s: float) -> Generator[RewardSignal, None, None]:
        video_frames = [await self._queue_video_frames.get()]
        print("reward func got first video frame")
        audio_frames = [await self._queue_audio_frames.get()]
        print("reward func got first audio frame")
        timestamp_last = min(video_frames[0].timestamp_s, audio_frames[0].timestamp_s)

        # TODO(TK): the logic of combining the two streams (at different frequencies) into a single stream of a constant
        #  frequency should be tested
        while True:
            # Keep sampling the lagging sensor stream:
            # TODO(TK): probably do prediction at the same time and perform in parallel with multiprocessing
            if audio_frames[-1].timestamp_s < video_frames[-1].timestamp_s:
                audio_frames.append(await self._queue_audio_frames.get())
                print("reward func got audio frame")
            else:
                video_frames.append(await self._queue_video_frames.get())
                print("reward func got video frame")

            # Are we ready to release a reward signal?
            timestamp_current = min(video_frames[-1].timestamp_s, audio_frames[-1].timestamp_s)
            if timestamp_current - timestamp_last >= period_s:
                included_video_frames = [frame for frame in video_frames if frame.timestamp_s < timestamp_current]
                included_audio_frames = [frame for frame in audio_frames if frame.timestamp_s < timestamp_current]

                # Emotion predictions:
                video_frame_predictions = [{key: value for emotion in face['proba_list'] for key, value in emotion.items()}
                                           for frame in included_video_frames
                                           for face in self._video_classifier.detect_emotion_for_single_frame(frame=frame.video_data)]
                audio_frame_predictions = [self._audio_classifier.predict_proba(audio_data=frame.audio_data, sample_rate=frame.sample_rate)
                                           for frame in included_audio_frames]

                df_video_emotions = pd.DataFrame(data=video_frame_predictions)
                df_audio_emotions = pd.DataFrame(data=audio_frame_predictions)

                # Calculate the combined reward:
                # TODO(TK): move these to a parameters file
                wt_audio = 1.0
                wt_video = 1.0
                series_audio_coefficients = pd.Series(data={
                    'happy': 1.0,
                    'neutral': -0.1,
                    'sad': -1.0,
                })
                series_video_coefficients = pd.Series(data={
                    'angry': -1.0,
                    'disgust': -1.0,
                    'fear': -1.0,
                    'happy': 1.0,
                    'sad': -1.0,
                    'surprise': 0.0,
                    'neutral': 0.0,
                })
                if not df_audio_emotions.empty and set(series_audio_coefficients.index) != set(df_audio_emotions.columns):
                    raise ValueError(f"Unexpected audio emotions: got {df_audio_emotions.columns} expected {series_audio_coefficients.index}")
                if not df_video_emotions.empty and set(series_video_coefficients.index) != set(df_video_emotions.columns):
                    raise ValueError(f"Unexpected video emotions: got {df_video_emotions.columns} expected {series_video_coefficients.index}")
                audio_reward = None if df_audio_emotions.empty else (series_audio_coefficients * df_audio_emotions).to_numpy().mean()
                video_reward = None if df_video_emotions.empty else (series_video_coefficients * df_video_emotions).to_numpy().mean()
                combined_reward = wt_audio * (audio_reward if audio_reward is not None else 0.0) + wt_video * (video_reward if video_reward is not None else 0.0)

                yield RewardSignal(
                    timestamp_s=timestamp_current,
                    combined_reward=combined_reward,
                    video_reward=video_reward,
                    audio_reward=audio_reward,
                    detected_video_emotions=df_video_emotions,
                    detected_audio_emotions=df_audio_emotions,
                )

                video_frames = [frame for frame in video_frames if frame.timestamp_s >= timestamp_current]
                audio_frames = [frame for frame in audio_frames if frame.timestamp_s >= timestamp_current]
                timestamp_last = timestamp_current


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--audio_period_propn', type=float, default=0.25)
    parser.add_argument('--audio_segment_duration_s', type=float, default=1.0)
    parser.add_argument('--reward_period_s', type=float, default=0.5)
    args = parser.parse_args()

    is_running = True

    def signal_handler(signum: Any, frame: Any) -> None:
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    if args.file is not None:
        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(file=args.file)
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(file=args.file)
    else:
        _audio_frame_generator = MicrophoneFrameGenerator()
        _video_frame_generator = WebcamFrameGenerator()

    with _audio_frame_generator as audio_frame_generator, _video_frame_generator as video_frame_generator:
        gen_video_frames = video_frame_generator.gen()
        gen_audio_frames = audio_frame_generator.gen(segment_duration_s=args.audio_segment_duration_s, period_propn=args.audio_period_propn)
        gen_sensors = interleave_temporally(
            GeneratorMeta(generator=gen_video_frames, get_timestamp=lambda video_frame: video_frame.timestamp_s),
            GeneratorMeta(generator=gen_audio_frames, get_timestamp=lambda audio_frame: audio_frame.timestamp_s),
        )

        reward_function = RewardFunction()
        gen_reward_signal = reward_function.gen(period_s=args.reward_period_s)

        generators = [gen_sensors, gen_reward_signal]
        tasks = [asyncio.create_task(generator.__anext__()) for generator in generators]

        while True:
            if len(tasks) == 0:
                return

            try:
                # Wait for the first task to complete:
                await next(asyncio.as_completed(tasks))

                results = []
                for idx in range(len(tasks)):
                    if tasks[idx].done():
                        results.append(tasks[idx].result())
                        tasks[idx] = asyncio.create_task(generators[idx].__anext__())
            except StopAsyncIteration:
                # Drop any done and StopAsyncException-throwing tasks and associated generators, as they're done
                idx_to_drop = [idx for idx in range(len(tasks)) if tasks[idx].done() and isinstance(tasks[idx].exception(), StopAsyncIteration)]
                for idx in reversed(idx_to_drop):
                    del tasks[idx]
                    del generators[idx]

                continue

            for result in results:
                if isinstance(result, VideoFrame):
                    print(f"Got video frame - timestamp={result.timestamp_s}")
                    await reward_function.push_video_frame(video_frame=result)
                elif isinstance(result, AudioFrame):
                    print(f"Got audio frame - timestamp={result.timestamp_s}")
                    await reward_function.push_audio_frame(audio_frame=result)
                elif isinstance(result, RewardSignal):
                    print(result)
                else:
                    raise RuntimeError()


if __name__ == '__main__':
    asyncio.run(main())
