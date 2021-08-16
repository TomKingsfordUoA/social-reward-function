# type: ignore

import asyncio

from social_robotics_reward.util import interleave_fifo, TaggedItem


def test_interleave_fifo() -> None:
    async def generator_0():
        x = 0.0
        while True:
            yield {'timestamp': x, 'x': x}
            x += 1

    async def generator_1():
        y = 0.0
        while True:
            yield {'timestamp': y + 1.5, 'y': y}
            y += 1

    async def accumulate():
        combined_generator = interleave_fifo({
            '0': generator_0(),
            '1': generator_1(),
        })

        return [await combined_generator.__anext__() for _ in range(5)]

    result = asyncio.run(accumulate())
    assert result == [
        TaggedItem(tags=('0',), item={'timestamp': 0.0, 'x': 0.0}),
        TaggedItem(tags=('1',), item={'timestamp': 1.5, 'y': 0.0}),
        TaggedItem(tags=('0',), item={'timestamp': 1.0, 'x': 1.0}),
        TaggedItem(tags=('1',), item={'timestamp': 2.5, 'y': 1.0}),
        TaggedItem(tags=('0',), item={'timestamp': 2.0, 'x': 2.0}),
    ]


def test_interleave_fifo_stop_at_first() -> None:
    async def generator_0():
        x = 0.0
        for _ in range(2):
            yield {'timestamp': x, 'x': x}
            x += 1

    async def generator_1():
        y = 0.0
        for _ in range(5):
            yield {'timestamp': y + 1.5, 'y': y}
            y += 1

    async def accumulate():
        combined_generator = interleave_fifo({'0': generator_0(), '1': generator_1()}, stop_at_first=True)

        return [elem async for elem in combined_generator]

    result = asyncio.run(accumulate())
    assert result == [
        TaggedItem(tags=('0',), item={'timestamp': 0.0, 'x': 0.0}),
        TaggedItem(tags=('1',), item={'timestamp': 1.5, 'y': 0.0}),
        TaggedItem(tags=('0',), item={'timestamp': 1.0, 'x': 1.0}),
        TaggedItem(tags=('1',), item={'timestamp': 2.5, 'y': 1.0}),
    ]


def test_interleave_fifo_stop_at_last() -> None:
    async def generator_0():
        x = 0.0
        for _ in range(2):
            yield {'timestamp': x, 'x': x}
            x += 1

    async def generator_1():
        y = 0.0
        for _ in range(5):
            yield {'timestamp': y + 1.5, 'y': y}
            y += 1

    async def accumulate():
        combined_generator = interleave_fifo({'0': generator_0(), '1': generator_1()}, stop_at_first=False)

        return [elem async for elem in combined_generator]

    result = asyncio.run(accumulate())
    assert result == [
        TaggedItem(tags=('0',), item={'timestamp': 0.0, 'x': 0.0}),
        TaggedItem(tags=('1',), item={'timestamp': 1.5, 'y': 0.0}),
        TaggedItem(tags=('0',), item={'timestamp': 1.0, 'x': 1.0}),
        TaggedItem(tags=('1',), item={'timestamp': 2.5, 'y': 1.0}),
        TaggedItem(tags=('1',), item={'timestamp': 3.5, 'y': 2.0}),
        TaggedItem(tags=('1',), item={'timestamp': 4.5, 'y': 3.0}),
        TaggedItem(tags=('1',), item={'timestamp': 5.5, 'y': 4.0}),
    ]
