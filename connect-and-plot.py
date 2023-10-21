#!/usr/bin/env python
import datetime
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from bleak import BleakClient, BleakScanner
from bokeh.document import without_document_lock
from bokeh.events import ButtonClick
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource
from bokeh.plotting import curdoc, figure

# TODO add a queue to store data
ROLLOVER = 200

i = 0
service_uuid = os.environ["SERVICE_UUID"]
device_uuid = os.environ["DEVICE_UUID"]
source_x = ColumnDataSource(data=dict(x=[0], y=[0]))
source_y = ColumnDataSource(data=dict(x=[0], y=[0]))
source_z = ColumnDataSource(data=dict(x=[0], y=[0]))
source_b = ColumnDataSource(data=dict(x=[0], y=[0]))

doc = curdoc()

executor = ThreadPoolExecutor(max_workers=2)

data_queue = queue.SimpleQueue()


async def set_title(title):
    p1.title.text = title


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


def read_queue_and_save_data():
    filename = f"accelerometer-data-{datetime.datetime.now()}.csv"
    with open(filename, "w") as f:
        f.write("timestamp,x,y,z,button_state\n")
    counter = 0
    while True:
        try:
            timestamp, data = data_queue.get_nowait()
            with open(filename, "a") as f:
                print(timestamp, *data, sep=",", file=f)
            counter += 1
        except queue.Empty:
            print(f"Queue empty, read {counter} items")
            return


@without_document_lock
async def read_data_from_device(device):
    async with BleakClient(device.address) as client:
        while client.is_connected:
            bytes_data = await client.read_gatt_char(service_uuid)
            timestamp = time.time()
            decoded_data = np.frombuffer(bytes_data, dtype=np.float32)
            data_queue.put((timestamp, decoded_data))

            doc.add_next_tick_callback(
                partial(
                    update,
                    x=decoded_data[0],
                    y=decoded_data[1],
                    z=decoded_data[2],
                    button_state=decoded_data[3],
                )
            )


async def update(x, y, z, button_state):
    # todo find a way to get index better than global counter
    global i
    i += 1

    # TODO make rollover an argument
    source_x.stream(
        new_data={
            "x": [source_x.data["x"][-1] + 1],
            "y": [x],
        },
        rollover=ROLLOVER,
    )
    source_y.stream(
        new_data={
            "x": [source_y.data["x"][-1] + 1],
            "y": [y],
        },
        rollover=ROLLOVER,
    )
    source_z.stream(
        new_data={
            "x": [source_z.data["x"][-1] + 1],
            "y": [z],
        },
        rollover=ROLLOVER,
    )
    source_b.stream(
        new_data={
            "x": [source_b.data["x"][-1] + 1],
            "y": [button_state],
        },
        rollover=ROLLOVER,
    )


p1 = figure(y_range=[-4, 4])
p1.title.text = "..."
lx = p1.line(x="x", y="y", source=source_x, color="red")
ly = p1.line(x="x", y="y", source=source_y, color="green")
lz = p1.line(x="x", y="y", source=source_z, color="blue")

p2 = figure(y_range=[-0.1, 1.5])

lb = p2.line(x="x", y="y", source=source_b, color="blue")

button = Button(label="Save data to file")

button.on_event(ButtonClick, read_queue_and_save_data)

doc.add_root(column(row(p1, p2), button))
doc.add_next_tick_callback(discover_device)

# TODO add button to start recording to file
