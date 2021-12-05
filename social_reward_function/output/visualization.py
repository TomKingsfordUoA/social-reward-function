import functools
import logging
import math
import time
import typing

import cv2  # type: ignore
import matplotlib  # type: ignore
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage  # type: ignore

from social_reward_function.config import VisualizationOutputConfig
from social_reward_function.input.video import VideoFrame
from social_reward_function.reward_function import RewardSignal


class RewardSignalVisualizer:
    def __init__(self, config: VisualizationOutputConfig) -> None:
        self._config = config

        self.__logger = logging.getLogger(__name__)

        if self._config.display_plots:
            # Ensure frames are maximized:
            if matplotlib.get_backend() == 'TkAgg':
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())

            self._ax_reward = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2)
            self._ax_emotions_live = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
            self._ax_emotions_average = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

            plt.show(block=False)
            plt.gcf().canvas.flush_events()

            self._time_begin: typing.Optional[float] = None
            self._video_frame_counter = 0
            self._axes_image: typing.Optional[AxesImage] = None
            self._reward_signal: typing.List[RewardSignal] = []
            self._observed_emotions: typing.List[str] = []
            self._max_observed_audio_power = 5e-3
            self._max_observed_reward = 1.0
            self._min_observed_reward = -1.0

        # Moving average calculation:
        self._moving_average_window: typing.List[RewardSignal] = []

    def __enter__(self) -> 'RewardSignalVisualizer':
        self._time_begin = time.time()
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        pass

    def _draw(self) -> None:
        if self._config.display_plots:
            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.gcf().canvas.flush_events()

    def _draw_video_frame(self, video_frame: VideoFrame, title: str) -> None:
        if not self._config.display_video:
            return

        # TODO(TK): bring back types with numpy>=1.20
        displayable_image = np.copy(video_frame.video_data)  # type: ignore
        cv2.putText(displayable_image, f"{video_frame.timestamp_s:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(title, displayable_image)
        cv2.waitKey(1)  # milliseconds

    def draw_video_live(self, video_frame: VideoFrame) -> None:
        self._draw_video_frame(video_frame=video_frame, title='Video (live)')

    def draw_video_downsampled(self, video_frame: VideoFrame) -> None:
        self._draw_video_frame(video_frame=video_frame, title='Video (downsampled)')

    # TODO(TK): This doesn't need to be async
    async def draw_reward_signal(
            self,
            reward_signal: RewardSignal,
    ) -> None:
        """
        Append and draw the latest RewardSignal
        """

        if not self._config.display_plots:
            return

        if self._time_begin is None:
            raise ValueError(f"{RewardSignalVisualizer.draw_reward_signal.__name__} called outside context manager")

        # Manage moving average:
        self._moving_average_window.append(reward_signal)
        self._moving_average_window = [elem for elem in self._moving_average_window
                                       if reward_signal.timestamp_s - elem.timestamp_s <= self._config.moving_average_window_width_s]
        average_reward_signal: typing.Optional[RewardSignal]
        if len(self._moving_average_window) != 0:
            average_reward_signal = functools.reduce(RewardSignal.__add__, self._moving_average_window)
            average_reward_signal /= len(self._moving_average_window)
        else:
            average_reward_signal = None
        self.__logger.info("average_reward_signal", average_reward_signal)

        lag = time.time() - self._time_begin - reward_signal.timestamp_s
        if lag > self._config.threshold_lag_s:
            self.__logger.info(f"reward signal viz falling behind! lag={lag:.2f}")

        # Append the new reward signal and drop old data points:
        self._reward_signal.append(reward_signal)
        self._reward_signal = [elem for elem in self._reward_signal if reward_signal.timestamp_s - elem.timestamp_s <= self._config.reward_window_width_s]

        color_combined = '#ff0000'
        color_audio = '#007700'
        color_video = '#000077'
        color_presence = '#777700'

        self._ax_reward.clear()
        self._ax_reward.axhline(y=0, color='k', alpha=0.5)  # x-axis
        if average_reward_signal is not None:
            self._ax_reward.axhline(y=average_reward_signal.combined_reward, color=color_combined, linestyle='--', alpha=0.5)
            if average_reward_signal.video_reward is not None:
                self._ax_reward.axhline(y=average_reward_signal.video_reward, color=color_video, linestyle='--', alpha=0.5)
            if average_reward_signal.audio_reward is not None:
                self._ax_reward.axhline(y=average_reward_signal.audio_reward, color=color_audio, linestyle='--', alpha=0.5)
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.combined_reward for elem in self._reward_signal],
            marker='x',
            color=color_combined,
            label='combined')
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.audio_reward for elem in self._reward_signal],
            marker='x',
            color=color_audio,
            label='audio')
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.video_reward for elem in self._reward_signal],
            marker='x',
            color=color_video,
            label='video')
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.presence_reward for elem in self._reward_signal],
            marker='x',
            color=color_presence,
            label='presence')

        timestamp_max = max(reward_signal.timestamp_s, self._config.reward_window_width_s)
        self._max_observed_reward = max(elem for elem in [
            self._max_observed_reward,
            np.max(reward_signal.combined_reward),  # type: ignore
            np.max(reward_signal.audio_reward) if reward_signal.audio_reward is not None else -math.inf,  # type: ignore
            np.max(reward_signal.video_reward) if reward_signal.video_reward is not None else -math.inf,  # type: ignore
            np.max(reward_signal.presence_reward) if reward_signal.presence_reward is not None else -math.inf,  # type: ignore
        ] if elem is not None)
        self._min_observed_reward = min(elem for elem in [
            self._min_observed_reward,
            np.min(reward_signal.combined_reward),  # type: ignore
            np.min(reward_signal.audio_reward) if reward_signal.audio_reward is not None else math.inf,  # type: ignore
            np.min(reward_signal.video_reward) if reward_signal.video_reward is not None else math.inf,  # type: ignore
            np.min(reward_signal.presence_reward) if reward_signal.presence_reward is not None else math.inf,  # type: ignore
        ] if elem is not None)

        self._ax_reward.set_xlim(left=timestamp_max - self._config.reward_window_width_s, right=time.time() - self._time_begin)
        self._ax_reward.set_ylim(bottom=self._min_observed_reward, top=self._max_observed_reward)
        self._ax_reward.set_title('Reward')
        self._ax_reward.set_ylabel('reward')
        self._ax_reward.set_xlabel('time')
        self._ax_reward.legend(loc='lower left')

        mean_detected_video_emotions_this_frame = reward_signal.detected_video_emotions.mean()
        mean_detected_audio_emotions_this_frame = reward_signal.detected_audio_emotions.mean()
        if average_reward_signal is not None:
            # TODO(TK): use np.typing.ArrayLike when numpy>=1.20
            mean_detected_video_emotions_all_frames: typing.Optional[typing.Any] = average_reward_signal.detected_video_emotions.mean()
            mean_detected_audio_emotions_all_frames: typing.Optional[typing.Any] = average_reward_signal.detected_audio_emotions.mean()
        else:
            mean_detected_video_emotions_all_frames = None
            mean_detected_audio_emotions_all_frames = None

        self._observed_emotions = sorted(
            set(self._observed_emotions) |
            set(mean_detected_video_emotions_this_frame.index) |
            set(mean_detected_audio_emotions_this_frame.index)
        )

        self._ax_emotions_live.clear()
        self._ax_emotions_live.bar(
            x=np.arange(len(self._observed_emotions)) - 0.25,
            height=[mean_detected_video_emotions_this_frame[emotion] if emotion in mean_detected_video_emotions_this_frame else 0.0
                    for emotion in self._observed_emotions],
            color=color_video,
            width=0.5,
            label='video',
        )
        self._ax_emotions_live.bar(
            x=np.arange(len(self._observed_emotions)) + 0.25,
            height=[mean_detected_audio_emotions_this_frame[emotion] if emotion in mean_detected_audio_emotions_this_frame else 0.0
                    for emotion in self._observed_emotions],
            color=color_audio,
            width=0.5,
            label='audio'
        )
        self._ax_emotions_live.set_ylim(bottom=0.0, top=1.0)
        self._ax_emotions_live.set_xticks(np.arange(len(self._observed_emotions)))
        self._ax_emotions_live.set_xticklabels(self._observed_emotions)
        self._ax_emotions_live.set_title('Detected Emotions (Live)')
        self._ax_emotions_live.legend(loc='upper right')

        self._ax_emotions_average.clear()
        if mean_detected_video_emotions_all_frames is not None:
            self._ax_emotions_average.bar(
                x=np.arange(len(self._observed_emotions)) - 0.25,
                height=[mean_detected_video_emotions_all_frames[emotion] if emotion in mean_detected_video_emotions_all_frames else 0.0
                        for emotion in self._observed_emotions],
                color=color_video,
                width=0.5,
                label='video',
            )
        if mean_detected_audio_emotions_all_frames is not None:
            self._ax_emotions_average.bar(
                x=np.arange(len(self._observed_emotions)) + 0.25,
                height=[mean_detected_audio_emotions_all_frames[emotion] if emotion in mean_detected_audio_emotions_all_frames else 0.0
                        for emotion in self._observed_emotions],
                color=color_audio,
                width=0.5,
                label='audio',
            )
        self._ax_emotions_average.set_ylim(bottom=0.0, top=1.0)
        self._ax_emotions_average.set_xticks(np.arange(len(self._observed_emotions)))
        self._ax_emotions_average.set_xticklabels(self._observed_emotions)
        self._ax_emotions_average.set_title('Detected Emotions (Moving Average)')
        self._ax_emotions_average.legend(loc='upper right')

        self._draw()

    def sustain(self) -> None:
        """
        Blocks and keeps the plot window open when data has finished
        """
        if self._time_begin is None:
            raise ValueError(f"{RewardSignalVisualizer.sustain.__name__} called outside context manager")

        plt.show()
