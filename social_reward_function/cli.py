import argparse
import asyncio
import logging
import pathlib
import signal
import sys
import time
import typing
from typing import Any

import pandas as pd  # type: ignore
import tqdm  # type: ignore
import yaml

from social_reward_function.config import Config, FileInputConfig
from social_reward_function.input.audio import AudioFrameGenerator, MicrophoneFrameGenerator, AudioFrame, \
    AudioFileFrameGenerator
from social_reward_function.input.video import VideoFrameGenerator, WebcamFrameGenerator, VideoFrame, \
    VideoFileFrameGenerator
from social_reward_function.output.file import RewardSignalFileWriter
from social_reward_function.output.visualization import RewardSignalVisualizer
from social_reward_function.reward_function import RewardFunction, RewardSignal
from social_reward_function.util import interleave_fifo, async_gen_callback_wrapper, TaggedItem


async def live(config: Config, logger: logging.Logger) -> None:
    if config.input.source.file is not None:
        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(
            file=config.input.source.file.path,
            segment_duration_s=config.input.audio.segment_duration_s,
            period_propn=config.input.audio.period_propn,
        )
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(
            target_fps=config.input.video.target_fps,
            config=config.input.source.file,
        )
    elif config.input.source.webcam is not None:
        _audio_frame_generator = MicrophoneFrameGenerator(
            segment_duration_s=config.input.audio.segment_duration_s,
            period_propn=config.input.audio.period_propn,
        )
        _video_frame_generator = WebcamFrameGenerator(
            target_fps=config.input.video.target_fps,
            config=config.input.source.webcam,
        )
    else:
        raise ValueError("Malformed config encountered")

    _reward_function = RewardFunction(config=config.reward_signal)
    _visualizer = RewardSignalVisualizer(config=config.output.visualization)
    _file_writer = RewardSignalFileWriter(config=config.output.file)

    with _audio_frame_generator as audio_frame_generator, \
            _video_frame_generator as video_frame_generator, \
            _reward_function as reward_function, \
            _visualizer as visualizer, \
            _file_writer as file_writer:

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
                # self.__logger.info(f"Got video frame (live) - timestamp={tagged_item.item.timestamp_s} "
                #       f"(wallclock={time.time() - time_begin})")  # TODO(TK): add at DEBUG logging level
                visualizer.draw_video_live(video_frame=tagged_item.item)

            elif _key_video_downsampled in tagged_item.tags:
                assert isinstance(tagged_item.item, VideoFrame)
                logger.info(f"Got video frame (downsampled) - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                reward_function.push_video_frame(video_frame=tagged_item.item)
                visualizer.draw_video_downsampled(video_frame=tagged_item.item)

            elif _key_audio in tagged_item.tags:
                assert isinstance(tagged_item.item, AudioFrame)
                logger.info(f"Got audio frame - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                reward_function.push_audio_frame(audio_frame=tagged_item.item)

            elif _key_reward in tagged_item.tags:
                assert isinstance(tagged_item.item, RewardSignal)
                logger.info(f"Got reward signal - timestamp={tagged_item.item.timestamp_s} (wallclock={time.time() - time_begin})")
                logger.info(tagged_item.item)
                await visualizer.draw_reward_signal(reward_signal=tagged_item.item)
                file_writer.append_reward_signal(reward_signal=tagged_item.item)

            else:
                raise RuntimeError(f"Unexpected {TaggedItem.__name__}: {tagged_item}")

        logger.info("stopped")
        visualizer.sustain()


async def dataset(config: Config, logger: logging.Logger) -> None:
    if config.input.source.dataset is None:
        raise RuntimeError('dataset() called with no dataset provided in config')
    dataset_path = pathlib.Path(config.input.source.dataset.path)

    output_dir = pathlib.Path(config.output.file.path)

    for video_path in tqdm.tqdm(sorted(dataset_path.glob('**/*.mp4')), file=sys.stdout):
        csv_path = output_dir.joinpath(video_path.relative_to(config.input.source.dataset.path)).parent.joinpath(f'{video_path.stem}.csv')
        if csv_path.is_file():
            logger.info(f'{str(csv_path)} already exists. Skipping...')
            continue
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        label = video_path.parent.stem
        reward_mapping = {
            'm2_strong_neg': -2,
            'm1_slight_neg': -1,
            '0_neutral': 0,
            '1_slight_pos': 1,
            '2_strong_pos': 2,
        }
        reward = reward_mapping[label]

        _audio_frame_generator: AudioFrameGenerator = AudioFileFrameGenerator(
            file=str(video_path),
            segment_duration_s=config.input.audio.segment_duration_s,
            period_propn=config.input.audio.period_propn,
        )
        _video_frame_generator: VideoFrameGenerator = VideoFileFrameGenerator(
            target_fps=config.input.video.target_fps,
            config=FileInputConfig(path=str(video_path), play_audio=False),
        )
        _reward_function = RewardFunction(config=config.reward_signal)

        _key_video_live = 'video_live'
        _key_video_downsampled = 'video_downsampled'
        _key_audio = 'audio'
        _key_sensors = 'sensors'
        _key_reward = 'reward'
        with _audio_frame_generator as audio_frame_generator, \
                _video_frame_generator as video_frame_generator, \
                _reward_function as reward_function:
            gen_sensors: typing.AsyncGenerator[TaggedItem[typing.Union[VideoFrame, AudioFrame]], None] = interleave_fifo(
                {
                    _key_video_live: video_frame_generator.gen_async_live(),
                    _key_video_downsampled: video_frame_generator.gen_async_downsampled(),
                    _key_audio: audio_frame_generator.gen_async(),
                },
                stop_at_first=False,
            )
            gen_sensors = async_gen_callback_wrapper(gen_sensors, callback_async=reward_function.stop_async())
            gen_combined: typing.AsyncGenerator[
                TaggedItem[typing.Union[VideoFrame, AudioFrame, RewardSignal]], None] = interleave_fifo(
                {
                    _key_sensors: typing.cast(
                        typing.AsyncGenerator[typing.Union[VideoFrame, AudioFrame, RewardSignal], None], gen_sensors),
                    _key_reward: reward_function.gen_async(),
                },
                stop_at_first=False,
            )

            reward_signals = []
            async for tagged_item in gen_combined:
                if _key_video_live in tagged_item.tags:
                    assert isinstance(tagged_item.item, VideoFrame)

                elif _key_video_downsampled in tagged_item.tags:
                    assert isinstance(tagged_item.item, VideoFrame)
                    reward_function.push_video_frame(video_frame=tagged_item.item)

                elif _key_audio in tagged_item.tags:
                    assert isinstance(tagged_item.item, AudioFrame)
                    reward_function.push_audio_frame(audio_frame=tagged_item.item)

                elif _key_reward in tagged_item.tags:
                    assert isinstance(tagged_item.item, RewardSignal)
                    reward_signals.append(tagged_item.item)

                else:
                    raise RuntimeError(f"Unexpected {TaggedItem.__name__}: {tagged_item}")

        df = pd.DataFrame(data=[
            dict(
                timestamp_s=reward_signal.timestamp_s,
                gt_reward=reward,
                combined_reward=reward_signal.combined_reward,
                audio_reward=reward_signal.audio_reward,
                video_reward=reward_signal.video_reward,
                presence_reward=reward_signal.presence_reward,
                human_detected=reward_signal.human_detected,
                **{f'audio_{key}': value for key, value in reward_signal.detected_audio_emotions.mean().to_dict().items()},
                **{f'video_{key}': value for key, value in reward_signal.detected_video_emotions.mean().to_dict().items()},
            )
            for reward_signal in reward_signals
        ])

        # Write to CSV:
        df.to_csv(
            str(csv_path),  # noqa
            header=True,
            index=True,
        )


def configure_logging() -> None:
    logging.captureWarnings(True)

    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    logger_root.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger_root.addHandler(ch)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(filename='log.log', mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger_root.addHandler(fh)


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='srf.yaml')
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    def signal_handler(signum: Any, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        with open(args.config) as f_config:
            dict_config = yaml.load(f_config, Loader=yaml.CLoader)
            config = Config.from_dict(dict_config)  # type: ignore

        if config.input.source.dataset is not None:
            await dataset(config=config, logger=logger)
        else:
            await live(config=config, logger=logger)
    except Exception as exc:
        logger.exception(exc)


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
