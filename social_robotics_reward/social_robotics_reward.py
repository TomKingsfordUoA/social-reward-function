import array
import os
import tempfile

import soundfile
import pyaudio
import wave


class MicrophoneSegmenter:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 1
    RATE = 44100  # sample rate

    def __init__(self):
        self._p = None
        self._stream = None

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()
        self._temp_dir.__exit__(exc_type, exc_val, exc_tb)

    def gen(self, segment_duration_s=2.0, period_propn=0.5):
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


if __name__ == '__main__':
    # Read audio data from file:
    audio_data, sample_rate = soundfile.read('samples/03-01-01-01-02-01-03_neutral.wav')
    print(audio_data.shape)
    print(sample_rate)

    # Read audio data from mic:
    with MicrophoneSegmenter() as microphone_segmenter:
        gen = microphone_segmenter.gen()
        l_audio_data = []
        for _ in range(5):
            audio_data, sample_rate = next(gen)
            l_audio_data.append(audio_data)

        print(len(audio_data))
