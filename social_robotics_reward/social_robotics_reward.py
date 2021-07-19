import array
import os
import tempfile

import soundfile
import pyaudio
import wave


def record_from_mic(segment_duration_s=4.0):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 1
    RATE = 44100  # sample rate

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)  # buffer

    frames = []
    for i in range(0, int(RATE / CHUNK * segment_duration_s)):
        frames.append(stream.read(CHUNK))  # 2 bytes(16 bits) per channel

    stream.stop_stream()
    stream.close()
    p.terminate()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'tmp.wav')

        # TODO(TK): ideally convert from frames to waveform without writing to file
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        return soundfile.read(temp_file)


if __name__ == '__main__':
    # Read audio data from file:
    audio_data, sample_rate = soundfile.read('samples/03-01-01-01-02-01-03_neutral.wav')
    print(audio_data.shape)
    print(sample_rate)

    # Read audio data from mic:
    audio_data, sample_rate = record_from_mic(segment_duration_s=2.0)
    print(audio_data.shape)
    print(sample_rate)
