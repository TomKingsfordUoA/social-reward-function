import json
import os
import random
import tempfile
import typing

import pandas as pd  # type: ignore
import pytest

from social_reward_function.config import FileOutputConfig
from social_reward_function.output.file import RewardSignalFileWriter
from social_reward_function.reward_function import RewardSignal


@pytest.fixture
def fake_reward_signal() -> typing.List[RewardSignal]:
    random.seed(42)
    return [
        RewardSignal(
            timestamp_s=0.5 * idx,
            combined_reward=random.random(),
            audio_reward=random.random(),
            video_reward=random.random(),
            presence_reward=random.random(),
            human_detected=random.random() > 0.5,
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
            path=str(tmp_dir),
            format='json',
            overwrite=True,
            enabled=True,
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
            format='yaml',
            overwrite=False,
            enabled=True,
        )
        with pytest.raises(FileExistsError):
            with RewardSignalFileWriter(config=config):
                pass


def test_write_reward_signal_summary_json(fake_reward_signal: typing.List[RewardSignal]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.json')
        config = FileOutputConfig(
            path=tmp_dir,
            format='json',
            overwrite=True,
            enabled=True,
        )
        assert not os.path.exists(output_file)
        with RewardSignalFileWriter(config=config) as file_writer:
            for reward_signal in fake_reward_signal:
                file_writer.append_reward_signal(reward_signal=reward_signal)

        with open(output_file) as f_output_file:
            json_output_file = json.load(f_output_file)
            assert json_output_file['summary'] == {
                'emotions': {
                    'mean_audio': {'c': 0.4051134271987294, 'd': 0.4420835460234761},
                    'mean_video': {'a': 0.5301972027333712, 'b': 0.5589190670523739}
                },
                'reward': {
                    'mean_audio': 0.10908787645810085,
                    'mean_combined': 0.44873410356246435,
                    'mean_video': 0.21290783908389888,
                    'mean_presence': 0.6691198974029255,
                }
            }


def test_write_reward_signal_summary_yaml(fake_reward_signal: typing.List[RewardSignal]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'output.yaml')
        config = FileOutputConfig(
            path=tmp_dir,
            format='yaml',
            overwrite=True,
            enabled=True,
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
                '      c: 0.4051134271987294\n' \
                '      d: 0.4420835460234761\n' \
                '    mean_video:\n' \
                '      a: 0.5301972027333712\n' \
                '      b: 0.5589190670523739\n' \
                '  reward:\n' \
                '    mean_audio: 0.10908787645810085\n' \
                '    mean_combined: 0.44873410356246435\n' \
                '    mean_presence: 0.6691198974029255\n' \
                '    mean_video: 0.21290783908389888\n'


def test_write_reward_signal_unrecognised_type() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FileOutputConfig(
            path=tmp_dir,
            format='yaml2',
            overwrite=True,
            enabled=True,
        )
        with pytest.raises(ValueError):
            RewardSignalFileWriter(config=config)
