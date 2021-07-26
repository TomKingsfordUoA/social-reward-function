#!/bin/env python3

import argparse
import asyncio
import signal
import sys
import time
from typing import Any, AsyncGenerator, Union, cast, Optional, List

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.image import AxesImage  # type: ignore

from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator
from social_robotics_reward.generator_coroutine_combiner import interleave_temporally, GeneratorMeta, interleave_fifo
from social_robotics_reward.reward_function import RewardFunction, RewardSignal
from social_robotics_reward.video_frame_generation import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator


class PlotDrawer:
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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--audio_period_propn', type=float, default=0.25)
    parser.add_argument('--audio_segment_duration_s', type=float, default=1.0)
    parser.add_argument('--reward_period_s', type=float, default=0.5)
    parser.add_argument('--video_frame_downsample_ratio', type=int, default=5)
    parser.add_argument('--reward_window_width', type=float, default=10.0)
    args = parser.parse_args()

    def signal_handler(signum: Any, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    if args.file is not None:
        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(file=args.file)
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(file=args.file)
    else:
        _audio_frame_generator = MicrophoneFrameGenerator()
        _video_frame_generator = WebcamFrameGenerator()

    with _audio_frame_generator as audio_frame_generator, _video_frame_generator as video_frame_generator:
        gen_video_frames = video_frame_generator.gen()
        gen_audio_frames = audio_frame_generator.gen(segment_duration_s=args.audio_segment_duration_s, period_propn=args.audio_period_propn)
        gen_sensors: AsyncGenerator[Union[VideoFrame, AudioFrame], None] = interleave_temporally([
            GeneratorMeta(generator=gen_video_frames, get_timestamp=lambda video_frame: cast(float, video_frame.timestamp_s)),
            GeneratorMeta(generator=gen_audio_frames, get_timestamp=lambda audio_frame: cast(float, audio_frame.timestamp_s)),
        ])

        plot_drawer = PlotDrawer(frame_downsample_ratio=args.video_frame_downsample_ratio, reward_window_width=args.reward_window_width)

        reward_function = RewardFunction()
        gen_reward_signal: AsyncGenerator[RewardSignal, None] = reward_function.gen(period_s=args.reward_period_s)

        # Interleave and stop when gen_sensors finishes (as gen_reward_signal will go forever):
        gen_combined = interleave_fifo([gen_sensors, gen_reward_signal], stop_at_first=True)

        async for result in gen_combined:
            if isinstance(result, VideoFrame):
                print(f"Got video frame - timestamp={result.timestamp_s}")
                plot_drawer.draw_video_frame(result)
                await reward_function.push_video_frame(video_frame=result)
            elif isinstance(result, AudioFrame):
                print(f"Got audio frame - timestamp={result.timestamp_s}")
                plot_drawer.draw_audio_frame(result)
                await reward_function.push_audio_frame(audio_frame=result)
            elif isinstance(result, RewardSignal):
                print(result)
                plot_drawer.draw_reward_signal(result)
            else:
                raise RuntimeError()


if __name__ == '__main__':
    asyncio.run(main())
