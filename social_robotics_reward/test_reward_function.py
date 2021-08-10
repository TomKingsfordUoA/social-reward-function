import pandas as pd

from social_robotics_reward.reward_function import RewardSignalConstants


def test_reward_function_constants_from_dict() -> None:
    d_constants = {
        'audio': {
            'overall': 0.5,
            'happy': 1.0,
            'neutral': 0.0,
            'sad': -1.0,
        },
        'video': {
            'overall': 0.4,
            'angry': -1.0,
            'disgust': -1.0,
            'fear': -1.0,
            'happy': 1.0,
            'sad': -1.0,
            'surprise': 0.0,
            'neutral': 0.0,
        }
    }
    reward_signal_constants = RewardSignalConstants.from_dict(d_constants)
    reward_signal_constants_expected = RewardSignalConstants(
        wt_audio_overall=0.5,
        wt_audio_happy=1.0,
        wt_audio_neutral=0.0,
        wt_audio_sad=-1.0,
        wt_video_overall=0.4,
        wt_video_angry=-1.0,
        wt_video_disgust=-1.0,
        wt_video_fear=-1.0,
        wt_video_happy=1.0,
        wt_video_sad=-1.0,
        wt_video_surprise=0.0,
        wt_video_neutral=0.0,
    )
    assert reward_signal_constants == reward_signal_constants_expected


def test_reward_function_constants_series() -> None:
    reward_signal_constants = RewardSignalConstants(
        wt_audio_overall=0.5,
        wt_audio_happy=1.0,
        wt_audio_neutral=0.0,
        wt_audio_sad=-1.0,
        wt_video_overall=0.4,
        wt_video_angry=-1.0,
        wt_video_disgust=-1.0,
        wt_video_fear=-1.0,
        wt_video_happy=1.0,
        wt_video_sad=-1.0,
        wt_video_surprise=0.0,
        wt_video_neutral=0.0,
    )
    assert reward_signal_constants.s_audio_coefficients.equals(pd.Series({
        'happy': 1.0,
        'neutral': 0.0,
        'sad': -1.0,
    }))
    assert reward_signal_constants.s_video_coefficients.equals(pd.Series({
        'angry': -1.0,
        'disgust': -1.0,
        'fear': -1.0,
        'happy': 1.0,
        'sad': -1.0,
        'surprise': 0.0,
        'neutral': 0.0,
    }))
