import asyncio
import dataclasses
import sys
import time
from typing import Optional, List, Set

import matplotlib  # type: ignore
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage  # type: ignore

from social_robotics_reward.reward_function import RewardSignal
from social_robotics_reward.sensors.audio import AudioFrame
from social_robotics_reward.sensors.video import VideoFrame


class RewardSignalVisualizer:
    def __init__(self, reward_window_width: float, video_downsample_rate: Optional[int]) -> None:
        self._reward_window_width = reward_window_width
        self._video_downsample_rate = video_downsample_rate

        # Ensure frames are maximized:
        if matplotlib.get_backend() == 'TkAgg':
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        self._ax_image = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
        self._ax_audio = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
        self._ax_reward = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=2)
        self._ax_emotions = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)

        plt.show(block=False)
        plt.gcf().canvas.flush_events()

        self._time_begin: Optional[float] = None
        self._video_frame_counter = 0
        self._axes_image: Optional[AxesImage] = None
        self._reward_signal: List[RewardSignal] = []
        self._previously_observed_video_emotions: Set[str] = set()
        self._previously_observed_audio_emotions: Set[str] = set()
        self._max_observed_audio_power = 5e-3

    async def _sync_time(self, timestamp_target: float, label: str) -> Optional[float]:
        """
        :return: None if not falling behind, otherwise the positive number of seconds behind target.
        """

        if self._time_begin is None:
            self._time_begin = time.time()

        time_wait = timestamp_target - (time.time() - self._time_begin)
        if time_wait >= 0:
            await asyncio.sleep(time_wait)
            return None
        else:
            print(f"{label} falling behind! {-time_wait}", file=sys.stderr)
            return -time_wait

    def _draw(self) -> None:
        plt.gcf().canvas.flush_events()
        plt.show(block=False)
        plt.gcf().canvas.flush_events()

    async def draw_video_frame(self, video_frame: VideoFrame) -> None:
        """
        Draw the latest VideoFrame
        """

        self._video_frame_counter += 1
        if self._video_downsample_rate is not None:
            if self._video_frame_counter % self._video_downsample_rate != 0:
                return

        falling_behind_s = await self._sync_time(timestamp_target=video_frame.timestamp_s, label='video')
        if falling_behind_s is not None and falling_behind_s >= 3.0:
            return

        if self._axes_image is None:
            self._axes_image = self._ax_image.imshow(video_frame.video_data)
        else:
            self._axes_image.set_data(video_frame.video_data)

        self._draw()

    async def draw_audio_frame(self, audio_frame: AudioFrame) -> None:
        """
        Draw the latest AudioFrame
        """

        power = np.power(audio_frame.audio_data, 2)
        t = np.linspace(start=0.0, stop=len(power) / audio_frame.sample_rate, num=len(power))
        max_power = float(np.max(power))  # type: ignore
        self._max_observed_audio_power = np.maximum(self._max_observed_audio_power, max_power)

        self._ax_audio.clear()
        self._ax_audio.plot(t, power)
        self._ax_audio.set_ylim(bottom=0, top=self._max_observed_audio_power)
        self._ax_audio.set_ylabel('power')
        self._ax_audio.set_xlabel('time')

        self._draw()

    async def draw_reward_signal(self, reward_signal: RewardSignal) -> None:
        """
        Append and draw the latest RewardSignal
        """

        if self._time_begin is None:
            self._time_begin = time.time()
        await self._sync_time(timestamp_target=reward_signal.timestamp_s, label='reward signal')

        # Append the new reward signal and drop old data points:
        self._reward_signal.append(reward_signal)
        self._reward_signal = [elem for elem in self._reward_signal if reward_signal.timestamp_s - elem.timestamp_s <= self._reward_window_width]

        color_combined = '#ff0000'
        color_audio = '#007700'
        color_video = '#000077'

        self._ax_reward.clear()
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

        timestamp_max = max(reward_signal.timestamp_s, self._reward_window_width)
        self._ax_reward.set_xlim(left=timestamp_max - self._reward_window_width, right=time.time() - self._time_begin)
        self._ax_reward.set_ylim(bottom=-2.0, top=2.0)
        self._ax_reward.set_title('Reward')
        self._ax_reward.set_ylabel('reward')
        self._ax_reward.set_xlabel('time')
        self._ax_reward.legend(loc='lower left')

        mean_detected_video_emotions = reward_signal.detected_video_emotions.mean()
        mean_detected_audio_emotions = reward_signal.detected_audio_emotions.mean()

        self._previously_observed_video_emotions |= set(mean_detected_video_emotions.index)
        self._previously_observed_audio_emotions |= set(mean_detected_audio_emotions.index)

        video_emotions = {
            'vid_' + emotion: mean_detected_video_emotions[emotion]
            for emotion in mean_detected_video_emotions.index
        }
        video_emotions.update({
            'vid_' + emotion: 0.0
            for emotion in self._previously_observed_video_emotions - set(mean_detected_video_emotions.index)
        })
        audio_emotions = {
            'aud_' + emotion: mean_detected_audio_emotions[emotion]
            for emotion in mean_detected_audio_emotions.index
        }
        audio_emotions.update({
            'aud_' + emotion: 0.0
            for emotion in self._previously_observed_audio_emotions - set(mean_detected_audio_emotions.index)
        })

        self._ax_emotions.clear()
        self._ax_emotions.bar(
            x=np.arange(len(video_emotions.keys())),
            height=video_emotions.values(),
            color=color_video,
        )
        self._ax_emotions.bar(
            x=len(video_emotions.keys()) + np.arange(len(audio_emotions.keys())),
            height=audio_emotions.values(),
            color=color_audio,
        )
        self._ax_emotions.set_xticks(np.arange(len(video_emotions.keys()) + len(audio_emotions.keys())))
        self._ax_emotions.set_xticklabels(list(video_emotions.keys()) + list(audio_emotions.keys()))
        self._ax_emotions.set_title('Detected Emotions')

        self._draw()

    def sustain(self) -> None:
        """
        Blocks and keeps the plot window open when data has finished
        """
        plt.show()
