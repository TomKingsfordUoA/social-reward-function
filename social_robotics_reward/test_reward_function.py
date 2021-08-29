import pandas as pd  # type: ignore

from social_robotics_reward.reward_function import RewardSignalConfig, EmotionWeights


def test_reward_function_constants_from_dict() -> None:
    d_constants = {
        'audio_weights': {
            'overall': 0.5,
            'angry': -1.0,
            'disgusted': -1.0,
            'fearful': -1.0,
            'happy': 1.0,
            'sad': -1.0,
            'surprised': 0.0,
            'neutral': 0.0,
        },
        'video_weights': {
            'overall': 0.4,
            'angry': -1.0,
            'disgusted': -1.0,
            'fearful': -1.0,
            'happy': 1.0,
            'sad': -1.0,
            'surprised': 0.0,
            'neutral': 0.0,
        },
        'period_s': 2.0,
        'threshold_audio_power': 0.01,
        'threshold_latency_s': 5.0,
    }
    reward_signal_constants = RewardSignalConfig.from_dict(d_constants)  # type: ignore
    reward_signal_constants_expected = RewardSignalConfig(
        audio_weights=EmotionWeights(
            overall=0.5,
            angry=-1.0,
            disgusted=-1.0,
            fearful=-1.0,
            happy=1.0,
            sad=-1.0,
            surprised=0.0,
            neutral=0.0,
        ),
        video_weights=EmotionWeights(
            overall=0.4,
            angry=-1.0,
            disgusted=-1.0,
            fearful=-1.0,
            happy=1.0,
            sad=-1.0,
            surprised=0.0,
            neutral=0.0,
        ),
        period_s=2.0,
        threshold_audio_power=0.01,
        threshold_latency_s=5.0,
    )
    assert reward_signal_constants == reward_signal_constants_expected


def test_reward_function_constants_series() -> None:
    reward_signal_constants = RewardSignalConfig(
        audio_weights=EmotionWeights(
            overall=0.5,
            angry=-1.0,
            disgusted=-1.0,
            fearful=-1.0,
            happy=1.0,
            sad=-1.0,
            surprised=0.0,
            neutral=0.0,
        ),
        video_weights=EmotionWeights(
            overall=0.4,
            angry=-1.0,
            disgusted=-1.0,
            fearful=-1.0,
            happy=1.0,
            sad=-1.0,
            surprised=0.0,
            neutral=0.0,
        ),
        period_s=2.0,
        threshold_audio_power=0.01,
        threshold_latency_s=5.0,
    )
    assert reward_signal_constants.s_audio_coefficients.equals(pd.Series({
        'angry': -1.0,
        'disgusted': -1.0,
        'fearful': -1.0,
        'happy': 1.0,
        'sad': -1.0,
        'surprised': 0.0,
        'neutral': 0.0,
    }))
    assert reward_signal_constants.s_video_coefficients.equals(pd.Series({
        'angry': -1.0,
        'disgusted': -1.0,
        'fearful': -1.0,
        'happy': 1.0,
        'sad': -1.0,
        'surprised': 0.0,
        'neutral': 0.0,
    }))
