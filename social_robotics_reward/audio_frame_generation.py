import abc
import asyncio
import dataclasses
import os
import tempfile
import time
import wave
from typing import Generator, Tuple, Any, Optional

import librosa  # type: ignore
import pyaudio  # type: ignore
import soundfile  # type: ignore
from numpy.typing import ArrayLike


@dataclasses.dataclass(frozen=True)
class AudioFrame:
    timestamp_s: float
    audio_data: ArrayLike
    sample_rate: int  # samples/sec


class AudioFrameGenerator(abc.ABC):
    def __enter__(self) -> 'AudioFrameGenerator':
        raise NotImplementedError()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        raise NotImplementedError()

    async def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[AudioFrame, None, None]:
        raise NotImplementedError()


# TODO(TK): optionally record the microphone audio to file for posterity
class MicrophoneFrameGenerator(AudioFrameGenerator):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 1
    RATE = 44100  # sample rate

    def __init__(self) -> None:
        self._p: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def __enter__(self) -> 'MicrophoneFrameGenerator':
        assert self._p is None
        assert self._stream is None

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=MicrophoneFrameGenerator.FORMAT,
            channels=MicrophoneFrameGenerator.CHANNELS,
            rate=MicrophoneFrameGenerator.RATE,
            input=True,
            frames_per_buffer=MicrophoneFrameGenerator.CHUNK)
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_dir.__enter__()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._p is not None:
            self._p.terminate()
        self._temp_dir.__exit__(exc_type, exc_val, exc_tb)

    async def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[AudioFrame, None, None]:
        if self._stream is None or self._p is None:
            raise NameError("Uninitialized. Did you forget to call inside a context manager?")

        chunks_per_segment = int(MicrophoneFrameGenerator.RATE / MicrophoneFrameGenerator.CHUNK * segment_duration_s)
        chunks_per_period = int(MicrophoneFrameGenerator.RATE / MicrophoneFrameGenerator.CHUNK * segment_duration_s * period_propn)

        frames = []
        time_initial = time.time()
        while True:
            # Each segment is a sequence of chunks. Read all the chunks for a new segment:
            for _ in range(chunks_per_segment):
                # TODO(TK): we should use non-blocking mode, have a callback populate a queue and await that queue being
                #  big enough
                frames.append(self._stream.read(MicrophoneFrameGenerator.CHUNK))  # 2 bytes(16 bits) per channel

            while len(frames) >= chunks_per_segment:
                # TODO(TK): ideally convert from frames to waveform without writing to file
                temp_file = os.path.join(self._temp_dir.name, 'tmp.wav')
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(MicrophoneFrameGenerator.CHANNELS)
                    wf.setsampwidth(self._p.get_sample_size(MicrophoneFrameGenerator.FORMAT))
                    wf.setframerate(MicrophoneFrameGenerator.RATE)
                    wf.writeframes(b''.join(frames[:chunks_per_segment]))

                audio_data, sample_rate = librosa.load(temp_file)
                yield AudioFrame(timestamp_s=time.time() - time_initial, audio_data=audio_data, sample_rate=sample_rate)

                # Advance a period:
                frames = frames[chunks_per_period:]


class AudioFileFrameGenerator(AudioFrameGenerator):
    def __init__(self, file: str) -> None:
        self._file = file

        self._audio_data, self._sample_rate = librosa.load(self._file)

    def __enter__(self) -> 'AudioFileFrameGenerator':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[AudioFrame, None, None]:
        segment_duration_samples = int(segment_duration_s * self._sample_rate)
        period_samples = int(segment_duration_s * self._sample_rate * period_propn)

        cursor = 0
        time_initial = time.time()
        while True:
            timestamp = (cursor + segment_duration_samples) / self._sample_rate

            # TODO(TK): this is necessary so video and audio produce at appropriate relative rates. Replace this with correct temporal mixing
            #  of multiple generators producing timstamped frames
            await asyncio.sleep(timestamp - (time.time() - time_initial))

            yield AudioFrame(timestamp_s=timestamp, audio_data=self._audio_data[cursor:cursor+segment_duration_samples], sample_rate=self._sample_rate)
            cursor += period_samples
            if cursor >= len(self._audio_data):
                return


if __name__ == '__main__':
    # Output dir:
    out_dir = 'out'
    os.mkdir(out_dir)

    # Read audio data from file:
    with AudioFileFrameGenerator(file='samples/01-01-01-01-01-01-01_neutral.mp4') as audio_file_segmenter:
        gen = audio_file_segmenter.gen(segment_duration_s=1.0, period_propn=0.25)
        l_audio_data = []
        for idx in range(10):
            try:
                audio_frame = asyncio.run(gen.__anext__())
                l_audio_data.append(audio_frame.audio_data)
                soundfile.write(os.path.join(out_dir, f'file_{idx}.wav'), audio_frame.audio_data, audio_frame.sample_rate)
            except StopIteration:
                break

    # Read audio data from mic:
    with MicrophoneFrameGenerator() as microphone_segmenter:
        gen = microphone_segmenter.gen()
        l_audio_data = []
        for idx in range(5):
            try:
                audio_frame = asyncio.run(gen.__anext__())
                l_audio_data.append(audio_frame.audio_data)
                soundfile.write(os.path.join(out_dir, f'mic_{idx}.wav'), audio_frame.audio_data, audio_frame.sample_rate)
            except StopIteration:
                break
