import json
import os
import random
import tempfile
import typing

import pandas as pd
import pytest
import yaml

from social_robotics_reward.output.file import FileOutputConfig, RewardSignalFileWriter
from social_robotics_reward.reward_function import RewardSignal


@pytest.fixture
def fake_reward_signal() -> typing.List[RewardSignal]:
    random.seed(42)
    return [
        RewardSignal(
            timestamp_s=0.5 * idx,
            combined_reward=random.random(),
            audio_reward=random.random(),
            video_reward=random.random(),
            detected_video_emotions=pd.DataFrame(data=[
                {'a': random.random(), 'b': random.random()} for _ in range(5)
            ]),
            detected_audio_emotions=pd.DataFrame(data=[
                {'c': random.random(), 'd': random.random()} for _ in range(5)
            ]),
        )
        for idx in range(3)
    ]


def test_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.json')
        config = FileOutputConfig(
            path=output_file,
            overwrite=True,
        )
        assert not os.path.exists(output_file)
        with RewardSignalFileWriter(config=config):
            pass
        assert os.path.exists(output_file)


def test_doesnt_overwrite_existing() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.json')

        # Create file:
        with open(output_file, 'w'):
            pass
        assert os.path.exists(output_file)

        config = FileOutputConfig(
            path=output_file,
            overwrite=False,
        )
        with pytest.raises(FileExistsError):
            with RewardSignalFileWriter(config=config):
                pass


def test_write_reward_signal_summary_json(fake_reward_signal: typing.List[RewardSignal]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.json')
        config = FileOutputConfig(
            path=output_file,
            overwrite=True,
        )
        assert not os.path.exists(output_file)
        with RewardSignalFileWriter(config=config) as file_writer:
            for reward_signal in fake_reward_signal:
                file_writer.append_reward_signal(reward_signal=reward_signal)

        with open(output_file) as f_output_file:
            json_output_file = json.load(f_output_file)
            assert json_output_file['summary'] == {
                'reward': {
                    'mean_combined': 0.2986359092264648,
                    'mean_video': 0.41576943591538207,
                    'mean_audio': 0.42006581017984673,
                },
                'emotions': {
                    'mean_video': {
                        'a': 0.46889227456526855,
                        'b': 0.4580541635040654,
                    },
                    'mean_audio': {
                        'c': 0.48354841541585536,
                        'd': 0.5015684367141928,
                    }
                }
            }


def test_write_reward_signal_summary_yaml(fake_reward_signal: typing.List[RewardSignal]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.yaml')
        config = FileOutputConfig(
            path=output_file,
            overwrite=True,
        )
        assert not os.path.exists(output_file)
        with RewardSignalFileWriter(config=config) as file_writer:
            for reward_signal in fake_reward_signal:
                file_writer.append_reward_signal(reward_signal=reward_signal)

        with open(output_file) as f_output_file:
            yaml_output_file = f_output_file.read()
            assert yaml_output_file == \
                'summary:\n' \
                '  emotions:\n' \
                '    mean_audio:\n' \
                '      c: 0.48354841541585536\n' \
                '      d: 0.5015684367141928\n' \
                '    mean_video:\n' \
                '      a: 0.46889227456526855\n' \
                '      b: 0.4580541635040654\n' \
                '  reward:\n' \
                '    mean_audio: 0.42006581017984673\n' \
                '    mean_combined: 0.2986359092264648\n' \
                '    mean_video: 0.41576943591538207\n'


def test_write_reward_signal_unrecognised_type() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.txt')
        config = FileOutputConfig(
            path=output_file,
            overwrite=True,
        )
        with pytest.raises(ValueError):
            RewardSignalFileWriter(config=config)
