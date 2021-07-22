import argparse
import signal
import typing
from typing import Any

import pandas as pd  # type: ignore

from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, AudioFileFrameGenerator, \
    MicrophoneFrameGenerator
from social_robotics_reward.external.emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer  # type: ignore
from social_robotics_reward.external.emotion_recognition_using_speech.test import get_estimators_name  # type: ignore
from social_robotics_reward.external.emotion_recognition_using_speech.utils import get_best_estimators  # type: ignore
from social_robotics_reward.external.residual_masking_network import RMN  # type: ignore
from social_robotics_reward.video_frame_generation import VideoFileFrameGenerator, VideoFrameGenerator, \
    WebcamFrameGenerator

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

    # Read audio segments:
    if args.audio:
        if args.file is not None:
            _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(file=args.file)
        else:
            _audio_frame_generator = MicrophoneFrameGenerator()
        voice_emotion_predictions = []
        with _audio_frame_generator as audio_frame_generator:
            for timestamp_s, audio_data, sample_rate in audio_frame_generator.gen(
                    segment_duration_s=1.0,
                    period_propn=0.25,
            ):
                if not is_running:
                    print("Stopping")
                    break

                # Predict audio segments
                estimators = get_best_estimators(classification=True)
                estimators_str, estimator_dict = get_estimators_name(estimators)
                best_estimator = max(estimators, key=lambda elem: typing.cast(float, elem[2]))  # elem[2] is the accuracy
                detector = EmotionRecognizer(best_estimator[0])

                predicted_emotions = {'timestamp': timestamp_s}
                predicted_emotions |= detector.predict_proba(audio_data=audio_data, sample_rate=sample_rate)
                voice_emotion_predictions.append(predicted_emotions)
            df_emotions = pd.DataFrame(voice_emotion_predictions).set_index('timestamp')
            print(df_emotions)
            print(df_emotions.mean())

    if args.video:
        if args.file is not None:
            _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(file=args.file)
        else:
            _video_frame_generator = WebcamFrameGenerator()
        facial_emotion_predictions = []
        with _video_frame_generator as video_frame_generator:
            rmn = RMN()

            for timestamp_s, video_frame in video_frame_generator.gen():
                if not is_running:
                    print("Stopping")
                    break

                # Predict video frames:
                detected_facial_emotions = rmn.detect_emotion_for_single_frame(video_frame)  # nb this is potentially multiple faces
                facial_emotion_predictions.extend([
                    {'timestamp': timestamp_s}
                    | {key: value for item in face['proba_list'] for key, value in item.items()} for face in detected_facial_emotions
                ])
                if len(facial_emotion_predictions) >= 1:
                    print(facial_emotion_predictions[-1])

            df_emotions = pd.DataFrame(facial_emotion_predictions).set_index('timestamp')
            print(df_emotions)
            print(df_emotions.mean())

    # TODO(TK): create a class which either reads from mic/webcam or from a video file, and produces a reward function
    #  at a specified periodicity (with the option of this periodicity being the video frame rate). The output should
    #  be timestamped so it can play at greater than realtime for a video, and it should just, at this period, calculate
    #  the reward based on the average of the as-yet-unused video/speech rewards
