from social_robotics_reward.audio_segment_generation import AudioFileSegmenter
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.emotion_recognition import \
    EmotionRecognizer
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.test import get_estimators_name
from social_robotics_reward.speech_emotion_recognition.emotion_recognition_using_speech.utils import get_best_estimators

if __name__ == '__main__':
    # Read audio segments:
    voice_emotion_predictions = []
    with AudioFileSegmenter(file='../samples/03-01-01-01-02-01-03_neutral.wav') as audio_segmenter:
        for audio_data, sample_rate in audio_segmenter.gen():
            try:
                # Predict audio segments
                estimators = get_best_estimators(classification=True)
                estimators_str, estimator_dict = get_estimators_name(estimators)
                detector = EmotionRecognizer(estimator_dict["BaggingClassifier"])
                voice_emotion_predictions.append(detector.predict_proba(audio_data=audio_data, sample_rate=sample_rate))
            except StopIteration:
                print('Audio finished')
                exit(0)
    for voice_emotion_prediction in voice_emotion_predictions:
        print(voice_emotion_prediction)

    # TODO(TK): read video frames
    # TODO(TK): predict video frames
