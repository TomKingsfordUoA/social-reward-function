import asyncio
import dataclasses
import time
from asyncio import Task
from datetime import timedelta
from typing import TypeVar, Callable, AsyncGenerator, List, Any, Optional, cast, Awaitable, Dict, Set, Tuple

T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class TaggedItem:
    tags: Tuple[str, ...]
    item: Any


async def interleave_fifo(generators: Dict[str, AsyncGenerator[Any, None]], stop_at_first: bool = False) -> AsyncGenerator[Any, None]:
    """
    Simply combined into an AsyncGenerator which yields the elements from the generators in the order they're yielded
    by the AsyncGenerators.
    """

    tasks: Dict[str, Optional[Task[Any]]] = {tag: asyncio.create_task(generator.__anext__()) for tag, generator in generators.items()}

    while True:
        remaining_tasks = [task for task in tasks.values() if task is not None]

        if (stop_at_first and (len(remaining_tasks) != len(generators))) or (not stop_at_first and (len(remaining_tasks) == 0)):
            for remaining_task in remaining_tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            if len(remaining_tasks) != 0:
                await asyncio.wait(remaining_tasks)
            for remaining_task in remaining_tasks:
                exc = remaining_task.exception()
                if exc is not None and not isinstance(exc, StopAsyncIteration):
                    raise exc
            return

        try:
            # Wait for the first task to complete:
            await asyncio.wait(remaining_tasks, return_when=asyncio.FIRST_COMPLETED)

            for tag, task in tasks.items():
                if task is not None and task.done():
                    result = task.result()
                    if isinstance(result, TaggedItem):
                        item = TaggedItem(tags=result.tags + (tag,), item=result.item)
                    else:
                        item = TaggedItem(tags=(tag,), item=result)
                    yield item
                    tasks[tag] = asyncio.create_task(generators[tag].__anext__())
        except StopAsyncIteration:
            # Drop any done and StopAsyncException-throwing tasks and associated generators, as they're done
            tags_to_drop = [tags for tags, task in tasks.items() if task is not None and task.done() and task.exception() is not None]
            for tags in tags_to_drop:
                exc = cast(Task[Any], tasks[tags]).exception()
                if exc is not None and not isinstance(exc, StopAsyncIteration):
                    raise exc
                tasks[tags] = None
            continue


async def async_gen_callback_wrapper(
        async_gen: AsyncGenerator[T, None],
        callback: Optional[Callable[[], None]] = None,
        callback_async: Optional[Awaitable[None]] = None,
) -> AsyncGenerator[T, None]:

    async for elem in async_gen:
        yield elem
    if callback_async is not None:
        await callback_async
    if callback is not None:
        callback()


class CodeBlockTimer:
    def __init__(self) -> None:
        self.__begin_time: Optional[float] = None
        self.__end_time: Optional[float] = None

    def __enter__(self) -> 'CodeBlockTimer':
        self.__begin_time = time.time()
        self.__end_time = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__end_time = time.time()

    @property
    def timedelta(self) -> timedelta:
        if self.__begin_time is None or self.__end_time is None:
            raise ValueError("Must be called after the context manager exits")
        return timedelta(seconds=self.__end_time - self.__begin_time)
