import asyncio
import dataclasses
from typing import TypeVar, Generic, Callable, AsyncGenerator, List

T = TypeVar('T')
Timestamp = float


@dataclasses.dataclass(frozen=True)
class GeneratorMeta(Generic[T]):
    generator: AsyncGenerator[T, None]
    get_timestamp: Callable[[T], Timestamp]


async def interleave_temporally(*generators: GeneratorMeta[T]) -> AsyncGenerator[T, None]:
    """
    Gets one element from each of the generators, then releases the element with the earliest timestamp in turn until
    all generators are exhausted.
    """

    remaining_generators = list(generators)
    funcs_get_timestamp = [generator_meta.get_timestamp for generator_meta in remaining_generators]

    elements: List[T] = [await generator_meta.generator.__anext__() for generator_meta in remaining_generators]
    timestamps: List[Timestamp] = [funcs_get_timestamp[idx](elements[idx]) for idx in range(len(elements))]

    while len(remaining_generators) != 0:
        releasable_element_idx = timestamps.index(min(timestamps))
        yield elements[releasable_element_idx]

        # Replace missing element:
        try:
            new_element = await remaining_generators[releasable_element_idx].generator.__anext__()
            elements[releasable_element_idx] = new_element
            timestamps[releasable_element_idx] = funcs_get_timestamp[releasable_element_idx](new_element)
        except StopAsyncIteration:
            del remaining_generators[releasable_element_idx]
            del funcs_get_timestamp[releasable_element_idx]
            del elements[releasable_element_idx]
            del timestamps[releasable_element_idx]


async def interleave_fifo(*generators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """
    Simply combined into an AsyncGenerator which yields the elements from the generators in the order they're yielded
    by the AsyncGenerators.
    """

    remaining_generators = list(generators)
    tasks = [asyncio.create_task(generator.__anext__()) for generator in remaining_generators]

    while True:
        if len(tasks) == 0:
            return

        try:
            # Wait for the first task to complete:
            await next(asyncio.as_completed(tasks))

            results = []
            for idx in range(len(tasks)):
                if tasks[idx].done():
                    yield tasks[idx].result()
                    tasks[idx] = asyncio.create_task(generators[idx].__anext__())
        except StopAsyncIteration:
            # Drop any done and StopAsyncException-throwing tasks and associated generators, as they're done
            idx_to_drop = [idx for idx in range(len(tasks)) if tasks[idx].done() and isinstance(tasks[idx].exception(), StopAsyncIteration)]
            for idx in reversed(idx_to_drop):
                del tasks[idx]
                del remaining_generators[idx]
            continue
