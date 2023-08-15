#!/usr/bin/env python

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from bokeh.document import without_document_lock
from bokeh.events import ButtonClick
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Button
from bokeh.plotting import curdoc, figure
import asyncio
from datetime import time, datetime


import numpy as np
from bleak import BleakScanner, BleakClient, BleakError

i = 0
service_uuid = "551de921-bbaa-4e0a-9374-3e30e88a9073"
device_uuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1"
source_x = ColumnDataSource(data=dict(x=[0], y=[0]))
source_y = ColumnDataSource(data=dict(x=[0], y=[0]))
source_z = ColumnDataSource(data=dict(x=[0], y=[0]))

doc = curdoc()

executor = ThreadPoolExecutor(max_workers=2)


async def set_title(title):
    p.title.text = title


@without_document_lock
async def discover_device():
    doc.add_next_tick_callback(partial(set_title, title="Scanning..."))
    device = None
    while device is None:
        print(f"Scanning for device with UUID {device_uuid}")
        devices = await BleakScanner.discover()
        selected_device = [d for d in devices if device_uuid in d.metadata["uuids"]]
        if not selected_device:
            print(f"Device with UUID {device_uuid} not found")
            doc.add_next_tick_callback(partial(set_title, title="Device not found"))
        else:
            print(f"Device with UUID {device_uuid} found")
            doc.add_next_tick_callback(partial(set_title, title="Device found"))
            device = selected_device[0]
            print(device)
            await read_data_from_device(device)


@without_document_lock
async def read_data_from_device(device):
    async with BleakClient(device.address) as client:
        while client.is_connected:
            bytes_data = await client.read_gatt_char(service_uuid)
            decoded_data = np.frombuffer(bytes_data, dtype=np.float32)
            print(decoded_data)
            doc.add_next_tick_callback(
                partial(
                    update,
                    x=decoded_data[0],
                    y=decoded_data[1],
                    z=decoded_data[2],
                )
            )


async def update(x, y, z):
    # todo find a way to get index better than global counter
    global i
    i += 1

    # TODO make rollover an argument
    source_x.stream(
        new_data={
            "x": [source_x.data["x"][-1] + 1],
            "y": [x],
        },
        rollover=100,
    )
    source_y.stream(
        new_data={
            "x": [source_y.data["x"][-1] + 1],
            "y": [y],
        },
        rollover=100,
    )
    source_z.stream(
        new_data={
            "x": [source_z.data["x"][-1] + 1],
            "y": [z],
        },
        rollover=100,
    )


p = figure(y_range=[-4, 4])
p.title.text = "..."
lx = p.line(x="x", y="y", source=source_x, color="red")
ly = p.line(x="x", y="y", source=source_y, color="green")
lz = p.line(x="x", y="y", source=source_z, color="blue")

doc.add_root(p)
doc.add_next_tick_callback(discover_device)
# TODO add button to start recording to file
