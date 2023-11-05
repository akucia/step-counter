#!/usr/bin/env python
import datetime
import os
import queue
from functools import partial
from pathlib import Path

from bleak import BleakScanner
from bokeh.document import without_document_lock
from bokeh.events import ButtonClick
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource
from bokeh.plotting import curdoc, figure

from step_counter.data_sources import BLESource, DummySource, MockSource, Source
from step_counter.models.logistic_regression.predict import LogisticRegressionPredictor

model_save_path = Path("models/logistic_regression.joblib")
model = LogisticRegressionPredictor(model_save_path)

PLOT_ROLLOVER = 200
# TODO change this global counter to something better
GLOBAL_PLOT_INDEX = 0

source_x = ColumnDataSource(data=dict(x=[0], y=[0]))
source_y = ColumnDataSource(data=dict(x=[0], y=[0]))
source_z = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button_score = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button_pred = ColumnDataSource(data=dict(x=[0], y=[0]))


doc = curdoc()
data_queue = queue.SimpleQueue()


async def set_title(title):
    p1.title.text = title


# TODO convert all of these 3 functions into one
@without_document_lock
async def use_ble_source():
    service_uuid = os.environ["SERVICE_UUID"]
    device_uuid = os.environ["DEVICE_UUID"]
    doc.add_next_tick_callback(partial(set_title, title="Scanning..."))
    source = None
    while source is None:
        print(f"Scanning for source with UUID {device_uuid}")
        devices = await BleakScanner.discover()
        selected_device = [d for d in devices if device_uuid in d.metadata["uuids"]]
        if not selected_device:
            print(f"Device with UUID {device_uuid} not found")
            doc.add_next_tick_callback(partial(set_title, title="Device not found"))
        else:
            print(f"Device with UUID {device_uuid} found")
            doc.add_next_tick_callback(partial(set_title, title="Device found"))
            source = BLESource(selected_device[0], service_uuid)
            print(source)
            await read_data_from_source(source)


@without_document_lock
async def use_dummy_source():
    doc.add_next_tick_callback(partial(set_title, title="Scanning..."))
    source = None
    while source is None:
        doc.add_next_tick_callback(partial(set_title, title="Device found"))
        source = DummySource()
        await read_data_from_source(source)


@without_document_lock
async def use_mock_source():
    doc.add_next_tick_callback(partial(set_title, title="Scanning..."))
    source = None
    while source is None:
        doc.add_next_tick_callback(partial(set_title, title="Device found"))
        source = MockSource(
            "data/steps/train/accelerometer-data-2023-08-25 20:31:42.558559.csv"
        )
        await read_data_from_source(source)


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
async def read_data_from_source(source: Source):
    async for timestamp, accelerometer_data, button_data in source.read_data():
        data_queue.put((timestamp, accelerometer_data))

        button_pred, button_pred_score = model.predict(
            accelerometer_data[0], accelerometer_data[1], accelerometer_data[2]
        )
        print(f"Button state: {button_data}, prediction: {button_pred_score:.3f}")

        doc.add_next_tick_callback(
            partial(
                update_plot,
                x=accelerometer_data[0],
                y=accelerometer_data[1],
                z=accelerometer_data[2],
                button_state=button_data,
                button_pred_score=button_pred_score,
                button_pred=button_pred,
            )
        )


async def _update_plot_source(source: ColumnDataSource, data: float):
    source.stream(
        new_data={
            "x": [source.data["x"][-1] + 1],
            "y": [data],
        },
        rollover=PLOT_ROLLOVER,
    )


async def update_plot(
    x: float,
    y: float,
    z: float,
    button_state: float,
    button_pred_score: float,
    button_pred: float,
):
    # todo find a way to get index better than global counter
    global GLOBAL_PLOT_INDEX
    GLOBAL_PLOT_INDEX += 1

    # TODO make rollover an argument
    await _update_plot_source(source_x, x)
    await _update_plot_source(source_y, y)
    await _update_plot_source(source_z, z)
    await _update_plot_source(source_button, button_state)
    await _update_plot_source(source_button_score, button_pred_score)
    await _update_plot_source(source_button_pred, button_pred)


p1 = figure(y_range=[-4, 4])
p1.title.text = "..."
lx = p1.line(x="x", y="y", source=source_x, color="red")
ly = p1.line(x="x", y="y", source=source_y, color="green")
lz = p1.line(x="x", y="y", source=source_z, color="blue")

p2 = figure(y_range=[-0.1, 1.5])

lb = p2.line(x="x", y="y", source=source_button, color="blue")
lb_pred = p2.line(x="x", y="y", source=source_button_pred, color="red")
p2.line(x="x", y="y", source=source_button_score, color="orange")

button = Button(label="Save data to file")

button.on_event(ButtonClick, read_queue_and_save_data)

doc.add_root(column(row(p1, p2), button))
doc.add_next_tick_callback(use_mock_source)

# TODO add button to start recording to file
