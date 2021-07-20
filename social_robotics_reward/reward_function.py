from social_robotics_reward.audio_segment_generation import AudioFileSegmenter

if __name__ == '__main__':
    # Read audio segments:
    with AudioFileSegmenter(file='samples/03-01-01-01-02-01-03_neutral.wav') as audio_segmenter:
        gen = audio_segmenter.gen()
        try:
            audio_data, sample_rate = next(gen)
        except StopIteration:
            print('Audio finished')
            exit(0)

    # TODO(TK): predict audio segments


    # TODO(TK): read video frames
    # TODO(TK): predicti video frames
    pass
