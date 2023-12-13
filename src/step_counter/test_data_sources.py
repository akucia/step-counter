import numpy as np
import pytest
from bleak import BLEDevice

from step_counter.data_sources import BLESource, DummySource, MockSource

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_read_data_random_source():
    source = DummySource()
    data = []
    async for timestamp, data_xyz, data_button in source.read_data():
        assert isinstance(timestamp, float)
        assert isinstance(data_xyz, np.ndarray)
        assert data_xyz.dtype == np.float32
        assert isinstance(data_button, float)
        assert len(data_xyz) == 3
        data.append((timestamp, data_xyz))
        if len(data) > 10:
            break
    assert len(data) == 11


@pytest.mark.asyncio
async def test_mock_source(dummy_train_and_test_data):
    train_path, _ = dummy_train_and_test_data
    source = MockSource(train_path / "train.csv")
    data = []
    async for timestamp, data_xyz, data_button in source.read_data():
        assert isinstance(timestamp, float)
        assert isinstance(data_xyz, np.ndarray)
        assert data_xyz.dtype == np.float32
        assert isinstance(data_button, float)
        assert len(data_xyz) == 3
        data.append((timestamp, data_xyz))
        if len(data) > 10:
            break
    assert len(data) == 11


@pytest.mark.asyncio
async def test_ble_source(mocker):
    # TODO fix this test
    #   AttributeError: 'NoneType' object has no attribute 'get_characteristic'

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
    data = []
    async for timestamp, data_xyz, data_button, *_ in source.read_data():
        assert isinstance(timestamp, float)
        assert isinstance(data_xyz, np.ndarray)
        assert data_xyz.dtype == np.float32
        assert isinstance(data_button, float)
        assert len(data_xyz) == 3
        data.append((timestamp, data_xyz))
        if len(data) > 10:
            break
    assert len(data) == 11
