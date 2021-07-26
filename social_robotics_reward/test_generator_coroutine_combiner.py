# type: ignore

import asyncio

from social_robotics_reward import generator_coroutine_combiner
from social_robotics_reward.generator_coroutine_combiner import GeneratorMeta


def test_interleave_temporally() -> None:
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
        combined_generator = generator_coroutine_combiner.interleave_temporally([
            GeneratorMeta(generator=generator_0(), get_timestamp=lambda d: d['timestamp']),
            GeneratorMeta(generator=generator_1(), get_timestamp=lambda d: d['timestamp']),
        ])

        return [await combined_generator.__anext__() for _ in range(5)]

    result = asyncio.run(accumulate())
    assert result == [
        {'timestamp': 0.0, 'x': 0.0},
        {'timestamp': 1.0, 'x': 1.0},
        {'timestamp': 1.5, 'y': 0.0},
        {'timestamp': 2.0, 'x': 2.0},
        {'timestamp': 2.5, 'y': 1.0},
    ]


def test_interleave_temporally_exhausted() -> None:
    class X:
        async def generator_0(self):
            x = 0.0
            for _ in range(2):
                yield {'timestamp': x, 'x': x}
                x += 1

        async def generator_1(self):
            y = 0.0
            while True:
                yield {'timestamp': y + 1.5, 'y': y}
                y += 1

    async def accumulate():
        combined_generator = generator_coroutine_combiner.interleave_temporally([
            GeneratorMeta(generator=X().generator_0(), get_timestamp=lambda d: d['timestamp']),
            GeneratorMeta(generator=X().generator_1(), get_timestamp=lambda d: d['timestamp']),
        ])

        return [await combined_generator.__anext__() for _ in range(5)]

    result = asyncio.run(accumulate())
    assert result == [
        {'timestamp': 0.0, 'x': 0.0},
        {'timestamp': 1.0, 'x': 1.0},
        {'timestamp': 1.5, 'y': 0.0},
        {'timestamp': 2.5, 'y': 1.0},
        {'timestamp': 3.5, 'y': 2.0},
    ]


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
        combined_generator = generator_coroutine_combiner.interleave_fifo([
            generator_0(),
            generator_1(),
        ])

        return [await combined_generator.__anext__() for _ in range(5)]

    result = asyncio.run(accumulate())
    assert result == [
        {'timestamp': 0.0, 'x': 0.0},
        {'timestamp': 1.5, 'y': 0.0},
        {'timestamp': 1.0, 'x': 1.0},
        {'timestamp': 2.5, 'y': 1.0},
        {'timestamp': 2.0, 'x': 2.0},
    ]


def test_interleave_fifo_stop_at_first() -> None:
    async def generator_0():
        x = 0.0
        for _ in range(2):
            yield {'timestamp': x, 'x': x}
            x += 1

    async def generator_1():
        y = 0.0
        while True:
            yield {'timestamp': y + 1.5, 'y': y}
            y += 1

    async def accumulate():
        combined_generator = generator_coroutine_combiner.interleave_fifo([generator_0(), generator_1()], stop_at_first=True)

        return [elem async for elem in combined_generator]

    result = asyncio.run(accumulate())
    assert result == [
        {'timestamp': 0.0, 'x': 0.0},
        {'timestamp': 1.5, 'y': 0.0},
        {'timestamp': 1.0, 'x': 1.0},
        {'timestamp': 2.5, 'y': 1.0},
    ]
