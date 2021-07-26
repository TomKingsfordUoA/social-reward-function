#!/bin/env python3

import argparse
import asyncio
import signal
from typing import Any, AsyncGenerator, Union, cast

from social_robotics_reward.audio_frame_generation import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator
from social_robotics_reward.generator_coroutine_combiner import interleave_temporally, GeneratorMeta, interleave_fifo
from social_robotics_reward.reward_function import RewardFunction, RewardSignal
from social_robotics_reward.video_frame_generation import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--audio_period_propn', type=float, default=0.25)
    parser.add_argument('--audio_segment_duration_s', type=float, default=1.0)
    parser.add_argument('--reward_period_s', type=float, default=0.5)
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

        reward_function = RewardFunction()
        gen_reward_signal: AsyncGenerator[RewardSignal, None] = reward_function.gen(period_s=args.reward_period_s)

        gen_combined = interleave_fifo([gen_sensors, gen_reward_signal])

        async for result in gen_combined:
            if isinstance(result, VideoFrame):
                print(f"Got video frame - timestamp={result.timestamp_s}")
                await reward_function.push_video_frame(video_frame=result)
            elif isinstance(result, AudioFrame):
                print(f"Got audio frame - timestamp={result.timestamp_s}")
                await reward_function.push_audio_frame(audio_frame=result)
            elif isinstance(result, RewardSignal):
                print(result)
            else:
                raise RuntimeError()


if __name__ == '__main__':
    asyncio.run(main())
