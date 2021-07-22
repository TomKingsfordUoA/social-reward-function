import abc
import os
from typing import Generator, Any

from numpy.typing import ArrayLike
import cv2  # type: ignore


class VideoFrameGenerator(abc.ABC):
    def gen(self) -> Generator[ArrayLike, None, None]:
        raise NotImplementedError()


class WebcamFrameGenerator(VideoFrameGenerator):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> 'WebcamFrameGenerator':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def gen(self) -> Generator[ArrayLike, None, None]:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
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

    def gen(self) -> Generator[ArrayLike, None, None]:
        cap = cv2.VideoCapture(self._file)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                return
        cap.release()
