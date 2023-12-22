import numpy as np
import pytest
from bleak import BLEDevice

from step_counter.data_sources import BLESource, DummySource, MockSource

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_read_data_random_source():
    source = DummySource()
    data_points = []
    async for data in source.read_data():
        assert isinstance(data.timestamp, float)
        assert isinstance(data.data_xyz, np.ndarray)
        assert data.data_xyz.dtype == np.float32
        assert isinstance(data.button_state, float)
        assert len(data.data_xyz) == 3
        data_points.append((data.timestamp, data.data_xyz))
        if len(data_points) > 10:
            break
    assert len(data_points) == 11


@pytest.mark.asyncio
async def test_mock_source(dummy_train_and_test_data):
    train_path, _ = dummy_train_and_test_data
    source = MockSource(train_path / "train.csv")
    data_points = []
    async for data in source.read_data():
        assert isinstance(data.timestamp, float)
        assert isinstance(data.data_xyz, np.ndarray)
        assert data.data_xyz.dtype == np.float32
        assert isinstance(data.button_state, float)
        assert len(data.data_xyz) == 3
        data_points.append((data.timestamp, data.data_xyz))
        if len(data_points) > 10:
            break
    assert len(data_points) == 11


@pytest.mark.asyncio
async def test_ble_source(mocker):
    mocker.patch("bleak.BleakScanner.discover", return_value=[])
    mocker.patch("bleak.BleakClient.connect", return_value=None)
    mocker.patch("bleak.BleakClient.is_connected", return_value=True)
    mocker.patch("bleak.BleakClient.start_notify", return_value=None)
    mocker.patch("bleak.BleakClient.stop_notify", return_value=None)
    mocker.patch("bleak.BleakClient.disconnect", return_value=None)
    mock_data = np.zeros(6, dtype=np.float32).tobytes()
    mocker.patch("bleak.BleakClient.read_gatt_char", return_value=mock_data)
    device = BLEDevice("00:00:00:00:00:00", "Test Device", 0, 0)
    source = BLESource(device, "00000000-0000-0000-0000-000000000000")
    data_points = []
    async for data in source.read_data():
        assert isinstance(data.timestamp, float)
        assert isinstance(data.data_xyz, np.ndarray)
        assert data.data_xyz.dtype == np.float32
        assert isinstance(data.button_state, float)
        assert len(data.data_xyz) == 3
        data_points.append((data.timestamp, data.data_xyz))
        if len(data_points) > 10:
            break
    assert len(data_points) == 11
