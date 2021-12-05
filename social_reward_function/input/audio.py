import abc
import asyncio
import dataclasses
import logging

import multiprocessing
import os
import queue
import tempfile
import time
import typing
import wave

import librosa  # type: ignore
import pyaudio  # type: ignore

from social_reward_function.util import CodeBlockTimer


@dataclasses.dataclass(frozen=True)
class AudioFrame:
    timestamp_s: float
    audio_data: typing.Any  # TODO(TK): replace with np.typing.ArrayLike when numpy upgrades to 1.20+ (conditional on TensorFlow support)
    sample_rate: int  # samples/sec


class AudioFrameGenerator(abc.ABC):
    def __init__(self, segment_duration_s: float, period_propn: float) -> None:
        self._segment_duration_s = segment_duration_s
        self._period_propn = period_propn

        self._queue: queue.Queue[AudioFrame] = multiprocessing.Queue()
        self._semaphore = multiprocessing.Semaphore(value=0)
        self._proc = multiprocessing.Process(target=self._gen)

    def __enter__(self) -> 'AudioFrameGenerator':
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        return

    def _start(self) -> None:
        return self._proc.start()

    def _gen(self) -> None:
        raise NotImplementedError()

    async def gen_async(self) -> typing.AsyncGenerator[AudioFrame, None]:
        try:
            if not self._proc.is_alive():
                self._proc.start()
            while self._proc.is_alive() or not self._queue.empty():
                if not self._semaphore.acquire(block=False):
                    await asyncio.sleep(0)
                    continue
                elem = self._queue.get()
                yield elem
            return
        except asyncio.CancelledError:
            pass


class MicrophoneFrameGenerator(AudioFrameGenerator):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 1
    RATE = 44100  # sample rate
    
    def __init__(self, segment_duration_s: float, period_propn: float):
        super().__init__(segment_duration_s, period_propn)
        self.__logger = logging.getLogger(__name__)

    def __enter__(self) -> 'MicrophoneFrameGenerator':
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_dir.__enter__()
        return self

    def _gen(self) -> None:
        _pyaudio = pyaudio.PyAudio()
        _stream = _pyaudio.open(
            format=MicrophoneFrameGenerator.FORMAT,
            channels=MicrophoneFrameGenerator.CHANNELS,
            rate=MicrophoneFrameGenerator.RATE,
            input=True,
            frames_per_buffer=MicrophoneFrameGenerator.CHUNK)

        chunks_per_segment = int(MicrophoneFrameGenerator.RATE / MicrophoneFrameGenerator.CHUNK * self._segment_duration_s)
        chunks_per_period = float(MicrophoneFrameGenerator.RATE / MicrophoneFrameGenerator.CHUNK * self._segment_duration_s * self._period_propn)

        frames = []
        time_initial = time.time()
        remainder_counter = 0.0
        try:
            while True:
                # Each segment is a sequence of chunks. Read all the chunks for a new segment:
                segment_timestamp = time.time() - time_initial
                self.__logger.info(f"Getting a fresh audio segment @ {segment_timestamp}")
                with CodeBlockTimer() as code_block_timer:
                    for _ in range(chunks_per_segment):
                        frames.append(_stream.read(MicrophoneFrameGenerator.CHUNK))  # 2 bytes(16 bits) per channel
                self.__logger.info(f"Fresh audio segment retrieval took {code_block_timer.timedelta}")

                with CodeBlockTimer() as code_block_timer:
                    while len(frames) >= chunks_per_segment:
                        # TODO(TK): ideally convert from frames to waveform without writing to file
                        temp_file = os.path.join(self._temp_dir.name, 'tmp.wav')
                        with wave.open(temp_file, 'wb') as wf:
                            wf.setnchannels(MicrophoneFrameGenerator.CHANNELS)
                            wf.setsampwidth(_pyaudio.get_sample_size(MicrophoneFrameGenerator.FORMAT))
                            wf.setframerate(MicrophoneFrameGenerator.RATE)
                            wf.writeframes(b''.join(frames[:chunks_per_segment]))

                        audio_data, sample_rate = librosa.load(temp_file)
                        timestamp = segment_timestamp - len(frames) / MicrophoneFrameGenerator.RATE * MicrophoneFrameGenerator.CHUNK

                        self._queue.put(AudioFrame(timestamp_s=timestamp, audio_data=audio_data, sample_rate=sample_rate))
                        self._semaphore.release()

                        # Advance a period:
                        chunks_per_period_whole = int(chunks_per_period)
                        chunks_per_period_remainder = chunks_per_period - chunks_per_period_whole
                        remainder_counter += chunks_per_period_remainder
                        advancement = chunks_per_period_whole + int(remainder_counter)
                        remainder_counter -= int(remainder_counter)
                        frames = frames[advancement:]
                self.__logger.info(f"Yielding audio segments took {code_block_timer.timedelta}")
        finally:
            _stream.stop_stream()
            _stream.close()
            _pyaudio.terminate()


class AudioFileFrameGenerator(AudioFrameGenerator):
    def __init__(self, file: str, segment_duration_s: float, period_propn: float) -> None:
        super().__init__(segment_duration_s, period_propn)
        self._file = file

        self._audio_data, self._sample_rate = librosa.load(self._file)

    def __enter__(self) -> 'AudioFileFrameGenerator':
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        return

    def _gen(self) -> None:
        segment_duration_samples = int(self._segment_duration_s * self._sample_rate)
        period_samples = int(self._segment_duration_s * self._sample_rate * self._period_propn)

        cursor = 0
        wallclock_begin = time.time()
        while True:
            timestamp_s = cursor / self._sample_rate

            wallclock_elapsed = time.time() - wallclock_begin
            audio_elapsed = timestamp_s
            wait_time = audio_elapsed - wallclock_elapsed
            if wait_time > 0:
                time.sleep(wait_time)

            self._queue.put(
                AudioFrame(
                    timestamp_s=timestamp_s,
                    audio_data=self._audio_data[cursor:cursor+segment_duration_samples],
                    sample_rate=self._sample_rate,
                )
            )
            self._semaphore.release()

            cursor += period_samples
            if cursor >= len(self._audio_data):
                return
