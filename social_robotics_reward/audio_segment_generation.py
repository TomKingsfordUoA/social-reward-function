import abc
import os
import tempfile
import wave
from typing import Generator, Tuple, Any, Optional

import pyaudio  # type: ignore
import soundfile  # type: ignore
from numpy.typing import ArrayLike


class AudioSegmenter(abc.ABC):
    def __enter__(self) -> 'AudioSegmenter':
        raise NotImplementedError()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        raise NotImplementedError()

    def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[Tuple[ArrayLike, int], None, None]:
        raise NotImplementedError()


class MicrophoneSegmenter(AudioSegmenter):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 1
    RATE = 44100  # sample rate

    def __init__(self) -> None:
        self._p: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def __enter__(self) -> 'MicrophoneSegmenter':
        assert self._p is None
        assert self._stream is None

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=MicrophoneSegmenter.FORMAT,
            channels=MicrophoneSegmenter.CHANNELS,
            rate=MicrophoneSegmenter.RATE,
            input=True,
            frames_per_buffer=MicrophoneSegmenter.CHUNK)
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

    def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[Tuple[ArrayLike, int], None, None]:
        if self._stream is None or self._p is None:
            raise NameError("Uninitialized. Did you forget to call inside a context manager?")

        chunks_per_segment = int(MicrophoneSegmenter.RATE / MicrophoneSegmenter.CHUNK * segment_duration_s)
        chunks_per_period = int(MicrophoneSegmenter.RATE / MicrophoneSegmenter.CHUNK * segment_duration_s * period_propn)

        frames = []
        while True:
            # Each segment is a sequence of chunks. Read all the chunks for a new segment:
            for _ in range(chunks_per_segment):
                frames.append(self._stream.read(MicrophoneSegmenter.CHUNK))  # 2 bytes(16 bits) per channel

            while len(frames) >= chunks_per_segment:
                # TODO(TK): ideally convert from frames to waveform without writing to file
                temp_file = os.path.join(self._temp_dir.name, 'tmp.wav')
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(MicrophoneSegmenter.CHANNELS)
                    wf.setsampwidth(self._p.get_sample_size(MicrophoneSegmenter.FORMAT))
                    wf.setframerate(MicrophoneSegmenter.RATE)
                    wf.writeframes(b''.join(frames[:chunks_per_segment]))

                yield soundfile.read(temp_file)

                # Advance a period:
                frames = frames[chunks_per_period:]


class AudioFileSegmenter(AudioSegmenter):
    def __init__(self, file: str) -> None:
        self._file = file

        self._audio_data, self._sample_rate = soundfile.read(self._file)

    def __enter__(self) -> 'AudioFileSegmenter':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def gen(self, segment_duration_s: float = 2.0, period_propn: float = 0.5) -> Generator[Tuple[ArrayLike, int], None, None]:
        segment_duration_samples = int(segment_duration_s * self._sample_rate)
        period_samples = int(segment_duration_s * self._sample_rate * period_propn)

        cursor = 0
        while True:
            yield self._audio_data[cursor:cursor+segment_duration_samples], self._sample_rate
            cursor += period_samples
            if cursor >= len(self._audio_data):
                return


if __name__ == '__main__':
    # Output dir:
    out_dir = 'out'
    os.mkdir(out_dir)  # raises if exists

    # Read audio data from file:
    with AudioFileSegmenter(file='samples/03-01-01-01-02-01-03_neutral.wav') as audio_file_segmenter:
        gen = audio_file_segmenter.gen()
        l_audio_data = []
        for idx in range(10):
            try:
                audio_data, sample_rate = next(gen)
                l_audio_data.append(audio_data)
                soundfile.write(os.path.join(out_dir, f'file_{idx}.wav'), audio_data, sample_rate)
            except StopIteration:
                break

    # Read audio data from mic:
    with MicrophoneSegmenter() as microphone_segmenter:
        gen = microphone_segmenter.gen()
        l_audio_data = []
        for idx in range(5):
            try:
                audio_data, sample_rate = next(gen)
                l_audio_data.append(audio_data)
                soundfile.write(os.path.join(out_dir, f'mic_{idx}.wav'), audio_data, sample_rate)
            except StopIteration:
                break
