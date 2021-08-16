import argparse
import asyncio
import dataclasses
import signal
import time
from typing import Any, AsyncGenerator, Dict, Optional

import cv2
import yaml

from social_robotics_reward.reward_function import RewardFunction, RewardSignal, RewardSignalConfig
from social_robotics_reward.sensors.audio import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator
from social_robotics_reward.sensors.video import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator
from social_robotics_reward.util import interleave_fifo, async_gen_callback_wrapper, TaggedItem
from social_robotics_reward.viz import RewardSignalVisualizer


@dataclasses.dataclass(frozen=True)
class Config:
    reward_signal_constants: RewardSignalConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Config':
        return Config(
            reward_signal_constants=RewardSignalConfig.from_dict(d['reward_signal']),
        )


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    # TODO(TK): move most of these to config file
    parser.add_argument('--viz_video_downsample_rate', type=int, default=1)
    parser.add_argument('--viz_reward_window_width', type=float, default=30.0)
    parser.add_argument('--viz_threshold_lag_s', type=float, default=10.0)
    parser.add_argument('--audio_period_propn', type=float, default=0.5)
    parser.add_argument('--audio_segment_duration_s', type=float, default=2.0)
    parser.add_argument('--video_target_fps', type=float, default=0.5)
    parser.add_argument('--config', type=str, default='srr.yaml')
    args = parser.parse_args()

    def signal_handler(signum: Any, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    with open(args.config) as f_config:
        config = Config.from_dict(yaml.load(f_config, Loader=yaml.CLoader))

    if args.file is not None:
        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(
            file=args.file,
            segment_duration_s=args.audio_segment_duration_s,
            period_propn=args.audio_period_propn,
        )
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(
            file=args.file,
            target_fps=args.video_target_fps,
        )
    else:
        _audio_frame_generator = MicrophoneFrameGenerator(
            segment_duration_s=args.audio_segment_duration_s,
            period_propn=args.audio_period_propn,
        )
        _video_frame_generator = WebcamFrameGenerator(
            target_fps=args.video_target_fps,
        )
    _reward_function = RewardFunction(config.reward_signal_constants)
    _plot_drawer = RewardSignalVisualizer(
        reward_window_width=args.viz_reward_window_width,
        video_downsample_rate=args.viz_video_downsample_rate,
        threshold_lag_s=args.viz_threshold_lag_s,
    )

    with _audio_frame_generator as audio_frame_generator, \
            _video_frame_generator as video_frame_generator, \
            _reward_function as reward_function, \
            _plot_drawer as plot_drawer:

        # Interleave and stop when gen_sensors finishes (as gen_reward_signal will go forever):
        _key_video_live = 'video_live'
        _key_video_downsampled = 'video_downsampled'
        _key_audio = 'audio'
        _key_sensors = 'sensors'
        _key_reward = 'reward'
        gen_sensors = interleave_fifo(
            {
                _key_video_live: video_frame_generator.gen_async_live(),
                _key_video_downsampled: video_frame_generator.gen_async_downsampled(),
                _key_audio: audio_frame_generator.gen_async(),
            },
            stop_at_first=False,
        )
        gen_sensors = async_gen_callback_wrapper(gen_sensors, callback_async=reward_function.stop_async())
        gen_combined = interleave_fifo(
            {
                _key_sensors: gen_sensors,
                _key_reward: reward_function.gen_async(),
            },
            stop_at_first=False,
        )

        time_begin = time.time()
        async for tagged_item in gen_combined:
            if _key_video_live in tagged_item.tags:
                assert isinstance(tagged_item.item, VideoFrame)
                # print(f"Got video frame (live) - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")  # TODO(TK): add at DEBUG logging level

                # Display image frame:
                displayable_image = tagged_item.item.video_data.copy()
                cv2.putText(displayable_image, f"{tagged_item.item.timestamp_s:.2f}", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Video (live)', displayable_image)
                cv2.waitKey(1)  # milliseconds

            elif _key_video_downsampled in tagged_item.tags:
                assert isinstance(tagged_item.item, VideoFrame)
                print(f"Got video frame (downsampled) - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                reward_function.push_video_frame(video_frame=tagged_item.item)

                # Display image frame:
                displayable_image = tagged_item.item.video_data.copy()
                cv2.putText(displayable_image, f"{tagged_item.item.timestamp_s:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Video (downsampled)', displayable_image)
                cv2.waitKey(1)  # milliseconds

            elif _key_audio in tagged_item.tags:
                assert isinstance(tagged_item.item, AudioFrame)
                print(f"Got audio frame - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                reward_function.push_audio_frame(audio_frame=tagged_item.item)

            elif _key_reward in tagged_item.tags:
                assert isinstance(tagged_item.item, RewardSignal)
                print(f"Got reward signal - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                print(tagged_item.item)
                await plot_drawer.draw_reward_signal(tagged_item.item)

            else:
                raise RuntimeError(f"Unexpected {TaggedItem.__name__}: {tagged_item}")

        print("stopped")
        plot_drawer.sustain()


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
