"""Accelerator data sources for step counter.

This module contains classes that can be used to read data from BLE device or mocked data sources
"""


import abc
import asyncio
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Union

import numpy as np
from bleak import BleakClient, BLEDevice


@dataclass(frozen=True)
class DeviceData:
    """Dataclass for storing data from the device."""

    timestamp: float
    data_xyz: np.ndarray
    button_state: float
    model_score: float
    model_prediction: float


class Source(abc.ABC):
    """Abstract base class for accelerator data sources."""

    @abc.abstractmethod
    def read_data(self) -> AsyncIterable[DeviceData]:
        """Read data from the source."""
        pass


class DummySource(Source):
    """Dummy data source that generates random data."""

    def __init__(self, seed: int = 42):
        """Initialize the random number generator.

        Args:
            seed: a seed for the random number generator

        """
        np.random.seed(seed)

    async def read_data(self) -> AsyncIterable[DeviceData]:
        """Generate random data indefinitely."""
        while True:
            await asyncio.sleep(0.05)
            timestamp = time.time()
            data_xyz = np.random.random(3).astype(np.float32)
            if np.random.random() > 0.9:
                data_button = 1.0
            else:
                data_button = 0.0
            yield DeviceData(timestamp, data_xyz, data_button, 0, 0)


class MockSource(Source):
    def __init__(self, file: Union[str, Path]):
        """Mock data source that reads data from a file.

        Args:
            file: a path to a csv file with data

        Notes:
            The file should have a header and the following columns:
            timestamp, x, y, z, button_state
        """
        self.file = file

    async def read_data(self) -> AsyncIterable[DeviceData]:
        """Read data from the file indefinitely."""

        with open(self.file) as f:
            all_lines = f.readlines()[1:]

        previous_timestamp, *_ = all_lines[1].split(",")
        previous_timestamp = float(previous_timestamp)
        # loop over the data indefinitely
        for line in itertools.cycle(all_lines):
            await asyncio.sleep(0.0)
            timestamp, x, y, z, button_state = line.split(",")
            timestamp = float(timestamp)
            data = np.array([float(x), float(y), float(z)])
            yield DeviceData(
                timestamp, data.astype(np.float32), float(button_state), 0, 0
            )
            time_diff = timestamp - previous_timestamp
            previous_timestamp = timestamp
            # pause exact amount of time between two timestamps read from file to simulate real-time data
            await asyncio.sleep(time_diff)


class BLESource(Source):
    """BLE data source that reads data from a BLE device."""

    def __init__(self, device: BLEDevice, service_uuid: str):
        self.device = device
        self.service_uuid = service_uuid

    async def read_data(self) -> AsyncIterable[DeviceData]:
        """Read data from the BLE device indefinitely."""
        async with BleakClient(self.device.address) as client:
            while client.is_connected:
                bytes_data = await client.read_gatt_char(self.service_uuid)
                timestamp = time.time()
                decoded_data = np.frombuffer(bytes_data, dtype=np.float32)
                yield DeviceData(
                    float(timestamp),
                    decoded_data[0:3],
                    float(decoded_data[3]),
                    float(decoded_data[4]),
                    float(decoded_data[5]),
                )
