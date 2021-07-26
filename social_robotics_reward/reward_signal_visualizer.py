import sys
import time
from typing import Optional, List

import matplotlib  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage  # type: ignore

from social_robotics_reward.audio_frame_generation import AudioFrame
from social_robotics_reward.reward_function import RewardSignal
from social_robotics_reward.video_frame_generation import VideoFrame


class RewardSignalVisualizer:
    def __init__(self, frame_downsample_ratio: int, reward_window_width: float) -> None:
        self._frame_downsample_ratio = frame_downsample_ratio
        self._reward_window_width = reward_window_width

        # Ensure frames are maximized:
        if matplotlib.get_backend() == 'TkAgg':
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        self._ax_image = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        self._ax_audio = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
        self._ax_reward = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=2)
        # TODO(TK): This should be a bar chart of the individual emotions detected in video and audio in the window producing
        #  the combined reward
        self._ax_detections = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)

        self._time_begin = time.time()
        self._video_frame_counter = 0
        self._axes_image: Optional[AxesImage] = None
        self._reward_signal: List[RewardSignal] = []

    def _sync_time(self, timestamp_target: float) -> None:
        time_wait = timestamp_target - (time.time() - self._time_begin)
        if time_wait >= 0:
            plt.pause(time_wait)
        else:
            plt.pause(1e-3)
            print("Falling behind!", file=sys.stderr)

    def draw_video_frame(self, video_frame: VideoFrame) -> None:
        """
        Draw the latest VideoFrame
        """

        if self._video_frame_counter == self._frame_downsample_ratio - 1:
            self._video_frame_counter = 0
            if self._axes_image is None:
                self._axes_image = self._ax_image.imshow(video_frame.video_data)
            else:
                self._axes_image.set_data(video_frame.video_data)

            self._sync_time(timestamp_target=video_frame.timestamp_s)
        else:
            self._video_frame_counter += 1

    def draw_audio_frame(self, audio_frame: AudioFrame) -> None:
        """
        Draw the latest AudioFrame
        """

        pass  # TODO(TK): implement

    def draw_reward_signal(self, reward_signal: RewardSignal) -> None:
        """
        Append and draw the latest RewardSignal
        """

        # Append the new reward signal and drop old data points:
        self._reward_signal.append(reward_signal)
        self._reward_signal = [elem for elem in self._reward_signal if reward_signal.timestamp_s - elem.timestamp_s <= self._reward_window_width]

        self._ax_reward.clear()
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.combined_reward for elem in self._reward_signal],
            color='#ff0000',
            label='combined')
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.audio_reward for elem in self._reward_signal],
            color='#007700',
            label='audio')
        self._ax_reward.plot(
            [elem.timestamp_s for elem in self._reward_signal],
            [elem.video_reward for elem in self._reward_signal],
            color='#000077',
            label='video')

        timestamp_max = max(reward_signal.timestamp_s, self._reward_window_width)
        self._ax_reward.set_xlim(left=timestamp_max - self._reward_window_width, right=timestamp_max)
        self._ax_reward.set_title('Reward')
        self._ax_reward.set_ylabel('reward')
        self._ax_reward.set_xlabel('time')
        self._ax_reward.legend(loc='lower left')

        self._sync_time(reward_signal.timestamp_s)
