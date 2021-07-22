import argparse
import signal
import typing
from typing import Any, Generator, Dict

from emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer
from emotion_recognition_using_speech.utils import get_best_estimators
from residual_masking_network import RMN
from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, AudioFileFrameGenerator, \
    MicrophoneFrameGenerator
from social_robotics_reward.video_frame_generation import VideoFileFrameGenerator, VideoFrameGenerator, \
    WebcamFrameGenerator


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
        self._speech_classifier = RewardFunction._load_speech_classifier()
        self._video_classifier = RMN()  # type: ignore

    @staticmethod
    def _load_speech_classifier() -> EmotionRecognizer:
        estimators = get_best_estimators(classification=True)  # type: ignore
        best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is accuracy
        return EmotionRecognizer(best_estimator[0])  # type: ignore

    # TODO(TK): This should actually return a reward signal (probably timestamp: float, overall: float, speech: float, facial: float)
    #  at a standard frequency (e.g. 30hz)
    def gen(self) -> Generator[Dict[str, float], None, None]:
        gen_video_frames = self._video_frame_generator.gen()
        gen_audio_frames = self._audio_frame_generator.gen(segment_duration_s=self._audio_segment_duration_s, period_propn=self._audio_period_propn)

        timestamp_s_video, frame_video = next(gen_video_frames)
        timestamp_s_audio, frame_audio, speech_sample_rate = next(gen_audio_frames)

        try:
            while True:
                if timestamp_s_audio < timestamp_s_video:
                    prediction_audio = self._speech_classifier.predict_proba(audio_data=frame_audio, sample_rate=speech_sample_rate)
                    yield {'timestamp': timestamp_s_audio} | prediction_audio

                    # Replace audio:
                    timestamp_s_audio, frame_audio, speech_sample_rate = next(gen_audio_frames)
                else:
                    prediction_video = self._video_classifier.detect_emotion_for_single_frame(frame_video)
                    for face in prediction_video:
                        yield {'timestamp': timestamp_s_video} | {key: value for item in face['proba_list'] for key, value in item.items()}

                    # Replace video:
                    timestamp_s_video, frame_video = next(gen_video_frames)
        except StopIteration:
            # When either generator reaches the end, the reward function ends
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--audio', action='store_true', default=True)
    parser.add_argument('--video', action='store_true', default=True)
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
            audio_period_propn=0.25,
            audio_segment_duration_s=1.0,
        )

        for reward in reward_function.gen():
            if not is_running:
                print("Stopping")
                break

            print(reward)

    # TODO(TK): create a class which either reads from mic/webcam or from a video file, and produces a reward function
    #  at a specified periodicity (with the option of this periodicity being the video frame rate). The output should
    #  be timestamped so it can play at greater than realtime for a video, and it should just, at this period, calculate
    #  the reward based on the average of the as-yet-unused video/speech rewards
