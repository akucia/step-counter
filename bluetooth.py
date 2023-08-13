#!/usr/bin/env python

import asyncio
from datetime import time, datetime

import matplotlib.pyplot as plt
from drawnow import drawnow

# set safe font for macos
plt.rcParams["font.family"] = "arial"
import numpy as np
from bleak import BleakScanner, BleakClient, BleakError

service_uuid = "551de921-bbaa-4e0a-9374-3e30e88a9073"
device_uuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1"

PLOT = False


def makeFig(data):
    """Make plot of accelerometer data.
    data: list of tuples of accelerometer data for 3 coordinates
    """
    data = np.array(data)
    plt.plot(data[:, 0], "ro-", label="X")
    plt.plot(data[:, 1], "go-", label="Y")
    plt.plot(data[:, 2], "bo-", label="Z")
    # plt.legend(loc="upper left")
    # plt.title("Accelerometer Data")
    plt.grid(True)
    plt.ylim(-4, 4)
    # plt.ylabel("Acceleration (G)")
    # plt.xlabel("Sample Number")


async def main():
    data = []
    while True:
        try:
            device = await discover_device(device_uuid)
            print(f"Found device {device.name}@{device.address}")
            filename = f"{device.name}-{device.address} - {datetime.now()}.csv"
            with open(filename, "w") as f:
                print("timestamp,x,y,z", file=f)

            async with BleakClient(device.address) as client:
                while True:
                    if client.is_connected:
                        bytes_data = await client.read_gatt_char(service_uuid)
                        decoded_data = np.frombuffer(bytes_data, dtype=np.float32)
                        print(decoded_data)
                        # write timestamp and decoded data to a csv file
                        with open(filename, "a") as f:
                            print(
                                f"{datetime.now().isoformat()},{decoded_data[0]},{decoded_data[1]},{decoded_data[2]}",
                                file=f,
                            )

                        if PLOT:
                            data.append(decoded_data)
                            drawnow(lambda: makeFig(data))
                            if len(data) > 50:
                                data.pop(0)
                    else:
                        print("Device disconnected")
                        break
        except BleakError as e:
            print(e)
            print("Trying again in 5 seconds...")
            await asyncio.sleep(5)


async def discover_device(device_uuid: str):
    device = None
    while device is None:
        # todo make these a command line argument with click
        devices = await BleakScanner.discover()
        selected_device = [d for d in devices if device_uuid in d.metadata["uuids"]]
        if not selected_device:
            print(f"Device with UUID {device_uuid} not found")
        else:
            device = selected_device[0]
    return device


if __name__ == "__main__":
    asyncio.run(main())
