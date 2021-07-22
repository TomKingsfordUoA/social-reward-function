import argparse
import signal
import typing
from typing import Any

import pandas as pd  # type: ignore

from social_robotics_reward.video_frame_generation import VideoFileFrameGenerator, VideoFrameGenerator, \
    WebcamFrameGenerator
from social_robotics_reward.audio_frame_generation import AudioFileFrameGenerator, MicrophoneFrameGenerator, AudioFrameGenerator
from social_robotics_reward.external.emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer  # type: ignore
from social_robotics_reward.external.emotion_recognition_using_speech.test import get_estimators_name  # type: ignore
from social_robotics_reward.external.emotion_recognition_using_speech.utils import get_best_estimators  # type: ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    args = parser.parse_args()

    is_running = True

    def signal_handler(signum: Any, frame: Any) -> None:
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    # # Read audio segments:
    # if args.file is not None:
    #     _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(file=args.file)
    # else:
    #     _audio_frame_generator = MicrophoneFrameGenerator()
    # voice_emotion_predictions = []
    # with _audio_frame_generator as audio_frame_generator:
    #     for audio_data, sample_rate in audio_frame_generator.gen(
    #             segment_duration_s=1.0,
    #             period_propn=0.5,
    #     ):
    #         if not is_running:
    #             print("Stopping")
    #             break
    #         try:
    #             # Predict audio segments
    #             estimators = get_best_estimators(classification=True)
    #             estimators_str, estimator_dict = get_estimators_name(estimators)
    #             best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is the accuracy
    #             detector = EmotionRecognizer(best_estimator[0])
    #             voice_emotion_predictions.append(detector.predict_proba(audio_data=audio_data, sample_rate=sample_rate))
    #         except StopIteration:
    #             print('Audio finished')
    #             exit(0)
    #     df_emotions = pd.DataFrame(voice_emotion_predictions)
    #     print(df_emotions)
    #     print(df_emotions.mean())

    if args.file is not None:
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(file=args.file)
    else:
        _video_frame_generator = WebcamFrameGenerator()
    facial_emotion_predictions = []
    with _video_frame_generator as video_frame_generator:
        for video_frame in video_frame_generator.gen():
            if not is_running:
                print("Stopping")
                break
            try:
                # Predict video frames:
                print(video_frame.shape)
                pass  # TODO(TK): implement
            except StopIteration:
                print('Video finished')
                exit(0)
        df_emotions = pd.DataFrame(facial_emotion_predictions)
        print(df_emotions)
        print(df_emotions.mean())

    # TODO(TK): create a class which either reads from mic/webcam or from a video file, and produces a reward function
    #  at a specified periodicity (with the option of this periodicity being the video frame rate). The output should
    #  be timestamped so it can play at greater than realtime for a video, and it should just, at this period, calculate
    #  the reward based on the average of the as-yet-unused video/speech rewards
