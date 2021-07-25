import argparse
import dataclasses
import signal
import typing
from typing import Any, Generator, Dict

import pandas as pd  # type: ignore

from emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer
from emotion_recognition_using_speech.utils import get_best_estimators
from residual_masking_network import RMN
from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, AudioFileFrameGenerator, \
    MicrophoneFrameGenerator
from social_robotics_reward.video_frame_generation import VideoFileFrameGenerator, VideoFrameGenerator, \
    WebcamFrameGenerator


@dataclasses.dataclass(frozen=True)
class RewardSignal:
    timestamp_s: float
    combined_reward: float
    detected_audio_emotions: pd.DataFrame
    detected_video_emotions: pd.DataFrame

    def __repr__(self) -> str:
        return f"RewardSignal(\n" \
               f"\ttimestamp_s={self.timestamp_s}\n" \
               f"\tcombined_reward={self.combined_reward}\n" \
               f"\tdetected_audio_emotions=\n{self.detected_audio_emotions.mean()}\n" \
               f"\tdetected_video_emotions=\n{self.detected_video_emotions.mean()}\n" \
               f")"


class RewardFunction:
    def __init__(
            self,
            video_frame_generator: VideoFrameGenerator,
            audio_frame_generator: AudioFrameGenerator,
            audio_segment_duration_s: float,
            audio_period_propn: float
    ) -> None:
        self._video_frame_generator = video_frame_generator
        self._audio_frame_generator = audio_frame_generator
        self._audio_segment_duration_s = audio_segment_duration_s
        self._audio_period_propn = audio_period_propn

        # Initialize emotion classifiers:
        self._audio_classifier = RewardFunction._load_audio_classifier()
        self._video_classifier = RMN()  # type: ignore

    @staticmethod
    def _load_audio_classifier() -> EmotionRecognizer:
        estimators = get_best_estimators(classification=True)  # type: ignore
        best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is accuracy
        return EmotionRecognizer(best_estimator[0])  # type: ignore

    # TODO(TK): This should probably return a RewardSignal object with timestamp, facial emotions, speech emotions, combined reward
    def gen(self, period_s: float) -> Generator[RewardSignal, None, None]:
        gen_video_frames = self._video_frame_generator.gen()
        gen_audio_frames = self._audio_frame_generator.gen(segment_duration_s=self._audio_segment_duration_s, period_propn=self._audio_period_propn)

        video_frames = [next(gen_video_frames)]
        audio_frames = [next(gen_audio_frames)]
        timestamp_last = min(video_frames[0].timestamp_s, audio_frames[0].timestamp_s)

        try:
            # TODO(TK): the logic of combining the two streams (at different frequencies) into a single stream of a constant
            #  frequency should be tested
            while True:
                # Keep sampling the lagging sensor stream:
                # TODO(TK): probably do prediction at the same time and perform in parallel with multiprocessing
                if audio_frames[-1].timestamp_s < video_frames[-1].timestamp_s:
                    audio_frames.append(next(gen_audio_frames))
                else:
                    video_frames.append(next(gen_video_frames))

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

                    yield RewardSignal(
                        timestamp_s=timestamp_current,
                        combined_reward=0.0,  # TODO(TK): implement
                        detected_video_emotions=pd.DataFrame(data=video_frame_predictions),
                        detected_audio_emotions=pd.DataFrame(data=audio_frame_predictions),
                    )

                    video_frames = [frame for frame in video_frames if frame.timestamp_s >= timestamp_current]
                    audio_frames = [frame for frame in audio_frames if frame.timestamp_s >= timestamp_current]
                    timestamp_last = timestamp_current

        except StopIteration:
            # When either generator reaches the end, the reward function ends
            return


if __name__ == '__main__':
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
        reward_function = RewardFunction(
            video_frame_generator=video_frame_generator,
            audio_frame_generator=audio_frame_generator,
            audio_period_propn=args.audio_period_propn,
            audio_segment_duration_s=args.audio_segment_duration_s,
        )

        for reward in reward_function.gen(period_s=args.reward_period_s):
            if not is_running:
                print("Stopping")
                break

            print(reward)
