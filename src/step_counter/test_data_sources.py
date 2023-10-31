import numpy as np
import pytest

from step_counter.data_sources import DummySource

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_read_data_random_source():
    source = DummySource()
    data = []
    async for timestamp, data_xyz in source.read_data():
        assert isinstance(timestamp, float)
        assert isinstance(data_xyz, np.ndarray)
        assert data_xyz.dtype == np.float64
        assert len(data_xyz) == 4
        data.append((timestamp, data_xyz))
        if len(data) > 10:
            break
    assert len(data) == 11
