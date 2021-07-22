import argparse
import signal

import pandas as pd

from social_robotics_reward.audio_segment_generation import AudioFileFrameGenerator, MicrophoneFrameGenerator
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.emotion_recognition import \
    EmotionRecognizer
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.test import get_estimators_name
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.utils import get_best_estimators

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    args = parser.parse_args()

    is_running = True

    def signal_handler(signum, frame):
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Read audio segments:
    if args.file is not None:
        _audio_segmenter = AudioFileFrameGenerator(file=args.file)
    else:
        _audio_segmenter = MicrophoneFrameGenerator()
    voice_emotion_predictions = []
    with _audio_segmenter as audio_segmenter:
        for audio_data, sample_rate in audio_segmenter.gen(
                segment_duration_s=1.0,
                period_propn=0.5,
        ):
            if not is_running:
                print("Stopping")
                break
            try:
                # Predict audio segments
                estimators = get_best_estimators(classification=True)
                estimators_str, estimator_dict = get_estimators_name(estimators)
                best_estimator = max(estimators, key=lambda elem: elem[2])  # elem[2] is the accuracy
                detector = EmotionRecognizer(best_estimator[0])
                voice_emotion_predictions.append(detector.predict_proba(audio_data=audio_data, sample_rate=sample_rate))
            except StopIteration:
                print('Audio finished')
                exit(0)
        df_emotions = pd.DataFrame(voice_emotion_predictions)
        print(df_emotions)
        print(df_emotions.mean())

    # TODO(TK): read video frames
    # TODO(TK): predict video frames

    # TODO(TK): create a class which either reads from mic/webcam or from a video file, and produces a reward function
    #  at a specified periodicity (with the option of this periodicity being the video frame rate). The output should
    #  be timestamped so it can play at greater than realtime for a video, and it should just, at this period, calculate
    #  the reward based on the average of the as-yet-unused video/speech rewards
