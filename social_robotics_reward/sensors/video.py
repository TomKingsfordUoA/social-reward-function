import abc
import asyncio
import dataclasses
import multiprocessing
import os
import queue
import time
from typing import Any, Optional, AsyncGenerator, Generator

import cv2  # type: ignore
from ffpyplayer.player import MediaPlayer
from numpy.typing import ArrayLike


@dataclasses.dataclass
class VideoFrame:
    timestamp_s: float
    video_data: ArrayLike


class VideoFrameGenerator(abc.ABC):
    def __init__(self, target_fps: float) -> None:
        self._target_fps = target_fps

        self._queue: queue.Queue[VideoFrame] = multiprocessing.Queue()
        self._semaphore = multiprocessing.Semaphore(value=0)
        self._proc = multiprocessing.Process(target=self._gen)

    def __enter__(self) -> 'VideoFrameGenerator':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return

    def _gen(self) -> None:
        raise NotImplementedError()

    def gen(self) -> Generator[VideoFrame, None, None]:
        if self._proc.is_alive():
            raise RuntimeError(f"{VideoFrameGenerator.__name__} already running")
        self._proc.start()
        while self._proc.is_alive() or not self._queue.empty():
            if not self._semaphore.acquire(timeout=1e-1):
                continue
            elem = self._queue.get()
            yield elem

    async def gen_async(self) -> AsyncGenerator[VideoFrame, None]:
        try:
            if self._proc.is_alive():
                raise RuntimeError(f"{VideoFrameGenerator.__name__} already running")
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


class WebcamFrameGenerator(VideoFrameGenerator):
    def _gen(self) -> None:
        cap = cv2.VideoCapture(0)  # noqa
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        timestamp_initial: Optional[float] = None
        timestamp_target: Optional[float] = None
        while True:
            ret, frame = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_initial is None:
                timestamp_initial = timestamp
            if timestamp_target is None:
                timestamp_target = timestamp

            if ret:
                if timestamp >= timestamp_target:
                    timestamp_target += 1.0 / self._target_fps
                    self._queue.put(VideoFrame(timestamp_s=timestamp - timestamp_initial, video_data=frame))
                    self._semaphore.release()
            else:
                cap.release()
                return


class VideoFileFrameGenerator(VideoFrameGenerator):
    def __init__(self, file: str, target_fps: float, play_audio: bool = True) -> None:
        super().__init__(target_fps=target_fps)
        self._file = file
        self._play_audio = play_audio

        if not os.path.exists(file):
            raise FileNotFoundError(file)

    def _gen(self) -> None:
        if self._play_audio:
            # we need to assign to a variable, even if unused, to prevent MediaPlayer from being GC'd
            audio_player = MediaPlayer(self._file)  # noqa
        cap = cv2.VideoCapture(self._file)  # noqa
        timestamp_target: Optional[float] = None
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file!")
        wallclock_begin = time.time()
        timestamp_begin: Optional[float] = None
        while True:
            ret, frame = cap.read()
            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # noqa
            if timestamp_begin is None:
                timestamp_begin = timestamp_s
            assert timestamp_begin == 0.0  # we can handle the case it isn't, but do expect it to be

            if ret:
                # TODO(TK): Why do we get some timestamp_s=0 frames at the end?
                if timestamp_target is None or timestamp_s >= timestamp_target:
                    timestamp_target = timestamp_target + 1.0 / self._target_fps if timestamp_target is not None else timestamp_s
                    self._queue.put(VideoFrame(timestamp_s=timestamp_s - timestamp_begin, video_data=frame))
                    self._semaphore.release()

                    wallclock_elapsed = time.time() - wallclock_begin
                    video_elapsed = timestamp_target - timestamp_begin
                    wait_time = video_elapsed - wallclock_elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

            else:
                cap.release()
                return


if __name__ == '__main__':
    # with VideoFileFrameGenerator(file='samples/01-01-03-01-02-01-01_happy.mp4') as video_frame_generator:
    with WebcamFrameGenerator(target_fps=25) as video_frame_generator:
        for video_frame in video_frame_generator.gen():
            print(video_frame)
