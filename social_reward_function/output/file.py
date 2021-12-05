import enum
import json
import os
import pathlib
import typing

import numpy as np
import pandas as pd  # type: ignore
import yaml

from social_reward_function.config import FileOutputConfig
from social_reward_function.reward_function import RewardSignal


class RewardSignalFileWriter:
    class OutputFileType(enum.Enum):
        JSON = 'json'
        YAML = 'yaml'

        @staticmethod
        def get_json_suffixes() -> typing.Iterable[str]:
            return {'json'}

        @staticmethod
        def get_yaml_suffixes() -> typing.Iterable[str]:
            return {'yaml', 'yml'}

    def __init__(self, config: FileOutputConfig) -> None:
        self._config = config

        # Performs checks on output file as early as possible:
        if self._config.enabled:
            self._output_file_type = self._get_output_file_type()
            if not self._config.overwrite and os.path.exists(self._config.path):
                raise FileExistsError(f'Output dir already exists: {self._config.path}')

        self._list_reward_signal: typing.List[RewardSignal] = []
        self._output_file = pathlib.Path(self._config.path).joinpath(f'output.{self._get_output_file_type().value}')
        self._f_output_file: typing.Optional[typing.TextIO] = None

    def __enter__(self) -> 'RewardSignalFileWriter':
        if self._config.enabled:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._f_output_file = self._output_file.open('w').__enter__()
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        try:
            if not self._config.enabled:
                return
            if self._f_output_file is None:
                raise RuntimeError("f_output_file is None! Did you call __exit__ before __enter__?")

            if self._output_file_type == RewardSignalFileWriter.OutputFileType.JSON:
                json.dump(self._generate_file_content(), self._f_output_file)
            elif self._output_file_type == RewardSignalFileWriter.OutputFileType.YAML:
                yaml.dump(self._generate_file_content(), self._f_output_file)
            else:
                raise ValueError(f"Unexpected suffix encountered in output file: {self._config.path}")
        finally:
            if self._f_output_file is not None:
                self._f_output_file.__exit__(exc_type, exc_val, exc_tb)

    def _get_output_file_type(self) -> OutputFileType:
        if self._config.format.lower() in RewardSignalFileWriter.OutputFileType.get_yaml_suffixes():
            return RewardSignalFileWriter.OutputFileType.YAML
        elif self._config.format.lower() in RewardSignalFileWriter.OutputFileType.get_json_suffixes():
            return RewardSignalFileWriter.OutputFileType.JSON
        else:
            raise ValueError(f"Unexpected suffix encountered in config: {self._config.format}")

    def _generate_file_content(self) -> typing.Dict[str, typing.Any]:
        return {
            'summary': {
                'reward': {
                    'mean_combined': float(np.mean([reward_signal.combined_reward for reward_signal in self._list_reward_signal])),
                    'mean_video': float(np.mean([reward_signal.video_reward for reward_signal in self._list_reward_signal
                                                 if reward_signal.video_reward is not None])),
                    'mean_audio': float(np.mean([reward_signal.audio_reward for reward_signal in self._list_reward_signal
                                                 if reward_signal.audio_reward is not None])),
                    'mean_presence': float(np.mean([reward_signal.presence_reward for reward_signal in self._list_reward_signal
                                                    if reward_signal.audio_reward is not None])),
                },
                'emotions': {
                    'mean_video': pd.concat([reward_signal.detected_video_emotions for reward_signal in self._list_reward_signal]).mean().to_dict()
                    if len(self._list_reward_signal) != 0 else {},
                    'mean_audio': pd.concat([reward_signal.detected_audio_emotions for reward_signal in self._list_reward_signal]).mean().to_dict()
                    if len(self._list_reward_signal) != 0 else {},
                }
            }
        }

    def append_reward_signal(self, reward_signal: RewardSignal) -> None:
        self._list_reward_signal.append(reward_signal)
