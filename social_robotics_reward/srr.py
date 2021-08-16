import argparse
import asyncio
import dataclasses
import signal
import time
from typing import Any, AsyncGenerator, Dict, Optional

import cv2
import yaml

from social_robotics_reward.reward_function import RewardFunction, RewardSignal, RewardSignalConstants
from social_robotics_reward.sensors.audio import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator
from social_robotics_reward.sensors.video import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator
from social_robotics_reward.util import interleave_fifo, async_gen_callback_wrapper
from social_robotics_reward.viz import RewardSignalVisualizer


@dataclasses.dataclass(frozen=True)
class Config:
    reward_signal_constants: RewardSignalConstants

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Config':
        return Config(
            reward_signal_constants=RewardSignalConstants.from_dict(d['reward_signal']),
        )


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    # TODO(TK): move most of these to config file
    parser.add_argument('--viz_video_downsample_rate', type=int, default=1)
    parser.add_argument('--viz_reward_window_width', type=float, default=30.0)
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
    )

    with _audio_frame_generator as audio_frame_generator, \
            _video_frame_generator as video_frame_generator, \
            _reward_function as reward_function, \
            _plot_drawer as plot_drawer:

        gen_video_frames = video_frame_generator.gen_async()
        gen_audio_frames = audio_frame_generator.gen_async()
        gen_reward_signal = reward_function.gen_async()

        # Interleave and stop when gen_sensors finishes (as gen_reward_signal will go forever):
        gen_sensors = interleave_fifo([gen_video_frames, gen_audio_frames], stop_at_first=False)
        gen_sensors = async_gen_callback_wrapper(gen_sensors, callback_async=reward_function.stop_async())
        gen_combined = interleave_fifo([gen_sensors, gen_reward_signal], stop_at_first=False)

        cnt_video_frames = 0.0
        cnt_audio_frames = 0.0
        time_begin = time.time()
        async for result in gen_combined:
            if isinstance(result, VideoFrame):
                cnt_video_frames += 1
                print(f"Got video frame - timestamp={result.timestamp_s} (wallclock={time.time() - time_begin})")
                if cnt_audio_frames != 0:
                    print(f"video_frames:audio_frames={cnt_video_frames / cnt_audio_frames}")  # useful for debugging lagging generators
                print(f"video fps={cnt_video_frames/(time.time() - time_begin)}")
                reward_function.push_video_frame(video_frame=result)

                # Display image frame:
                displayable_image = result.video_data.copy()
                cv2.putText(displayable_image, f"{result.timestamp_s:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('video', displayable_image)
                cv2.waitKey(1)

            elif isinstance(result, AudioFrame):
                cnt_audio_frames += 1
                print(f"Got audio frame - timestamp={result.timestamp_s} (wallclock={time.time() - time_begin})")
                if cnt_audio_frames != 0:
                    print(f"video_frames:audio_frames={cnt_video_frames / cnt_audio_frames}")  # useful for debugging lagging generators
                reward_function.push_audio_frame(audio_frame=result)
            elif isinstance(result, RewardSignal):
                print(f"Got reward signal - timestamp={result.timestamp_s} (wallclock={time.time() - time_begin})")
                print(result)
                await plot_drawer.draw_reward_signal(result)
            else:
                raise RuntimeError(f"Unexpected type: {type(result)}")

        print("stopped")
        plot_drawer.sustain()


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
