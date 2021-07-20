import os

from social_robotics_reward.audio_segment_generation import AudioFileSegmenter
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.emotion_recognition import \
    EmotionRecognizer
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.test import get_estimators_name
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.utils import get_best_estimators

if __name__ == '__main__':
    # Read audio segments:
    voice_emotion_predictions = []
    root_dir = '../samples'
    for file in os.listdir(root_dir):
        with AudioFileSegmenter(file=os.path.join(root_dir, file)) as audio_segmenter:
            for audio_data, sample_rate in audio_segmenter.gen(
                segment_duration_s=3.0,
                period_propn=0.25,
            ):
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
        print(file)
        for voice_emotion_prediction in voice_emotion_predictions:
            print('\t' + str(voice_emotion_prediction))

    # TODO(TK): read video frames
    # TODO(TK): predict video frames
