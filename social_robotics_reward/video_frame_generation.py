import abc
import asyncio
import dataclasses
import os
from typing import Generator, Any, Tuple, Optional
import time

from numpy.typing import ArrayLike
import cv2  # type: ignore


@dataclasses.dataclass
class VideoFrame:
    timestamp_s: float
    video_data: ArrayLike


# TODO(TK): these are expensive operations so should probably use multiprocessing
class VideoFrameGenerator(abc.ABC):
    def __enter__(self) -> 'VideoFrameGenerator':
        raise NotImplementedError()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        raise NotImplementedError()

    async def gen(self) -> Generator[VideoFrame, None, None]:
        raise NotImplementedError()


# TODO(TK): optionally record the webcam video to file for posterity
class WebcamFrameGenerator(VideoFrameGenerator):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> 'WebcamFrameGenerator':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def gen(self) -> Generator[VideoFrame, None, None]:
        cap = cv2.VideoCapture(0)  # noqa
        timestamp_initial = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield VideoFrame(timestamp_s=time.time() - timestamp_initial, video_data=frame)
            else:
                return
        cap.release()


class VideoFileFrameGenerator(VideoFrameGenerator):
    def __init__(self, file: str) -> None:
        self._file = file

        if not os.path.exists(file):
            raise FileNotFoundError(file)

    def __enter__(self) -> 'VideoFileFrameGenerator':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def gen(self) -> Generator[VideoFrame, None, None]:
        cap = cv2.VideoCapture(self._file)  # noqa
        timestamp_s_prev: Optional[float] = None
        time_initial = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # noqa

            if ret:
                # TODO(TK): Why do we get some timestamp_s=0 frames at the end?
                if timestamp_s_prev is None or timestamp_s > timestamp_s_prev:
                    # TODO(TK): this is necessary so video and audio produce at appropriate relative rates. Replace this with correct temporal mixing
                    #  of multiple generators producing timstamped frames
                    await asyncio.sleep(timestamp_s - (time.time() - time_initial))

                    yield VideoFrame(timestamp_s=timestamp_s, video_data=frame)
                timestamp_s_prev = timestamp_s
            else:
                return
        cap.release()
