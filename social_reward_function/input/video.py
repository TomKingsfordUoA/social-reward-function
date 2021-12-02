import abc
import asyncio
import dataclasses
import multiprocessing
import os
import queue
import time
import typing

import cv2  # type: ignore
from ffpyplayer.player import MediaPlayer  # type: ignore

from social_reward_function.config import WebcamInputConfig, FileInputConfig


@dataclasses.dataclass
class VideoFrame:
    timestamp_s: float
    video_data: typing.Any  # TODO(TK): replace with np.typing.ArrayLike when numpy upgrades to 1.20+ (conditional on TensorFlow support)


class VideoFrameGenerator(abc.ABC):
    def __init__(self, target_fps: float) -> None:
        self._target_fps = target_fps

        self._queue_live: queue.Queue[VideoFrame] = multiprocessing.Queue()
        self._semaphore_live = multiprocessing.Semaphore(value=0)
        self._queue_downsampled: queue.Queue[VideoFrame] = multiprocessing.Queue()
        self._semaphore_downsampled = multiprocessing.Semaphore(value=0)
        self._proc = multiprocessing.Process(target=self._gen)

    def __enter__(self) -> 'VideoFrameGenerator':
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        return

    def _gen(self) -> None:
        raise NotImplementedError()

    async def gen_async_live(self) -> typing.AsyncGenerator[VideoFrame, None]:
        try:
            if not self._proc.is_alive():
                self._proc.start()
            while self._proc.is_alive() or not self._queue_live.empty():
                if not self._semaphore_live.acquire(block=False):
                    await asyncio.sleep(0)
                    continue
                elem = self._queue_live.get()
                yield elem
            return
        except asyncio.CancelledError:
            pass

    async def gen_async_downsampled(self) -> typing.AsyncGenerator[VideoFrame, None]:
        try:
            if not self._proc.is_alive():
                self._proc.start()
            while self._proc.is_alive() or not self._queue_live.empty():
                if not self._semaphore_downsampled.acquire(block=False):
                    await asyncio.sleep(0)
                    continue
                elem = self._queue_downsampled.get()
                yield elem
            return
        except asyncio.CancelledError:
            pass


class WebcamFrameGenerator(VideoFrameGenerator):
    def __init__(self, target_fps: float, config: WebcamInputConfig) -> None:
        super().__init__(target_fps=target_fps)
        self._config = config

    def _gen(self) -> None:
        cap = cv2.VideoCapture(self._config.device_id)  # noqa
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        timestamp_initial: typing.Optional[float] = None
        timestamp_target: typing.Optional[float] = None
        while True:
            ret, frame = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_initial is None:
                timestamp_initial = timestamp
            if timestamp_target is None:
                timestamp_target = timestamp

            if ret:
                video_frame = VideoFrame(timestamp_s=timestamp - timestamp_initial, video_data=frame)
                self._queue_live.put(video_frame)
                self._semaphore_live.release()
                if timestamp >= timestamp_target:
                    timestamp_target += 1.0 / self._target_fps
                    self._queue_downsampled.put(video_frame)
                    self._semaphore_downsampled.release()
            else:
                cap.release()
                return


class VideoFileFrameGenerator(VideoFrameGenerator):
    def __init__(self, target_fps: float, config: FileInputConfig) -> None:
        super().__init__(target_fps=target_fps)
        self._config = config

        if not os.path.exists(self._config.path):
            raise FileNotFoundError(self._config.path)

    def _gen(self) -> None:
        if self._config.play_audio:
            # we need to assign to a variable, even if unused, to prevent MediaPlayer from being GC'd
            # TODO(TK): consider moving this to the audio file sensor
            audio_player = MediaPlayer(self._config.path)
            print("MediaPlayer (audio) loaded", flush=True)
        else:
            audio_player = None
        cap = cv2.VideoCapture(self._config.path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file!")
        print("VideoCapture file loaded", flush=True)

        timestamp_target: typing.Optional[float] = None
        wallclock_begin = time.time()
        timestamp_begin: typing.Optional[float] = None
        while True:
            # Retrieve a frame to precent garbage collection:
            if audio_player is not None:
                audio_player.get_frame(force_refresh=False, show=False)

            ret, frame = cap.read()
            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # noqa
            if timestamp_begin is None:
                timestamp_begin = timestamp_s
            assert timestamp_begin == 0.0  # we can handle the case it isn't, but do expect it to be

            if ret:
                video_frame = VideoFrame(timestamp_s=timestamp_s - timestamp_begin, video_data=frame)
                self._queue_live.put(video_frame)
                self._semaphore_live.release()
                # TODO(TK): Why do we get some timestamp_s=0 frames at the end?
                if timestamp_target is None or timestamp_s >= timestamp_target:
                    timestamp_target = timestamp_target + 1.0 / self._target_fps if timestamp_target is not None else timestamp_s
                    self._queue_downsampled.put(video_frame)
                    self._semaphore_downsampled.release()

                if timestamp_target is not None:
                    wallclock_elapsed = time.time() - wallclock_begin
                    video_elapsed = timestamp_s - timestamp_begin
                    wait_time = video_elapsed - wallclock_elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

            else:
                cap.release()
                return
