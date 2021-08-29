import argparse
import asyncio
import dataclasses
import signal
import time
import typing
from typing import Any

import dataclasses_json
import yaml

from social_robotics_reward.reward_function import RewardFunction, RewardSignal, RewardSignalConfig
from social_robotics_reward.sensors.audio import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator, AudioInputConfig
from social_robotics_reward.sensors.video import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator, VideoInputConfig, FileInputConfig, WebcamInputConfig
from social_robotics_reward.util import interleave_fifo, async_gen_callback_wrapper, TaggedItem
from social_robotics_reward.viz import RewardSignalVisualizer, RewardSignalVisualizerConstants


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class InputConfig:
    audio: AudioInputConfig
    video: VideoInputConfig
    file: typing.Optional[FileInputConfig] = dataclasses.field(default=None)
    webcam: typing.Optional[WebcamInputConfig] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        if not ((self.file is None) ^ (self.webcam is None)):
            raise ValueError("Exactly one of 'webcam' and 'file' must be specified")


@dataclasses_json.dataclass_json(undefined='raise')
@dataclasses.dataclass(frozen=True)
class Config:
    input: InputConfig
    reward_signal: RewardSignalConfig
    visualization: RewardSignalVisualizerConstants


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='srr.yaml')
    args = parser.parse_args()

    def signal_handler(signum: Any, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    with open(args.config) as f_config:
        dict_config = yaml.load(f_config, Loader=yaml.CLoader)
        config = Config.from_dict(dict_config)  # type: ignore

    if config.input.file is not None:
        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(
            file=config.input.file.path,
            segment_duration_s=config.input.audio.segment_duration_s,
            period_propn=config.input.audio.period_propn,
        )
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(
            target_fps=config.input.video.target_fps,
            config=config.input.file,
        )
    elif config.input.webcam is not None:
        _audio_frame_generator = MicrophoneFrameGenerator(
            segment_duration_s=config.input.audio.segment_duration_s,
            period_propn=config.input.audio.period_propn,
        )
        _video_frame_generator = WebcamFrameGenerator(
            target_fps=config.input.video.target_fps,
            config=config.input.webcam,
        )
    else:
        raise ValueError("Exactly one of config.input.file and config.input.webcam must be non-None")
    _reward_function = RewardFunction(config=config.reward_signal)
    _plot_drawer = RewardSignalVisualizer(config=config.visualization)

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
        gen_sensors: typing.AsyncGenerator[TaggedItem[typing.Union[VideoFrame, AudioFrame]], None] = interleave_fifo(
            {
                _key_video_live: video_frame_generator.gen_async_live(),
                _key_video_downsampled: video_frame_generator.gen_async_downsampled(),
                _key_audio: audio_frame_generator.gen_async(),
            },
            stop_at_first=False,
        )
        gen_sensors = async_gen_callback_wrapper(gen_sensors, callback_async=reward_function.stop_async())
        gen_combined: typing.AsyncGenerator[TaggedItem[typing.Union[VideoFrame, AudioFrame, RewardSignal]], None] = interleave_fifo(
            {
                _key_sensors: typing.cast(typing.AsyncGenerator[typing.Union[VideoFrame, AudioFrame, RewardSignal], None], gen_sensors),
                _key_reward: reward_function.gen_async(),
            },
            stop_at_first=False,
        )

        time_begin = time.time()
        async for tagged_item in gen_combined:
            if _key_video_live in tagged_item.tags:
                assert isinstance(tagged_item.item, VideoFrame)
                # print(f"Got video frame (live) - timestamp={tagged_item.item.timestamp_s} "
                #       f"(wallclock={time.time() - time_begin})")  # TODO(TK): add at DEBUG logging level
                plot_drawer.draw_video_live(video_frame=tagged_item.item)

            elif _key_video_downsampled in tagged_item.tags:
                assert isinstance(tagged_item.item, VideoFrame)
                print(f"Got video frame (downsampled) - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                reward_function.push_video_frame(video_frame=tagged_item.item)
                plot_drawer.draw_video_downsampled(video_frame=tagged_item.item)

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
