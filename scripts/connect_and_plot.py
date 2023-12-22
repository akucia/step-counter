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
from bokeh.models import Button, ColumnDataSource, Div
from bokeh.plotting import curdoc, figure

from step_counter.data_sources import (
    BLESource,
    DeviceData,
    DummySource,
    MockSource,
    Source,
)
from step_counter.models.dnn_keras.predict import TFLiteDNNPredictor

model_save_path = Path("models/dnn/tflite/model_quantized.tflite")
model = TFLiteDNNPredictor(model_save_path)

PLOT_ROLLOVER = 200
# TODO change this global counter to something better
GLOBAL_PLOT_INDEX = 0

source_x = ColumnDataSource(data=dict(x=[0], y=[0]))
source_y = ColumnDataSource(data=dict(x=[0], y=[0]))
source_z = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button_score = ColumnDataSource(data=dict(x=[0], y=[0]))
source_button_pred = ColumnDataSource(data=dict(x=[0], y=[0]))

button_clicks = Div(text="clicks: 0", render_as_text=True)
model_predictions = Div(text="predictions: 0", render_as_text=True)


doc = curdoc()
data_queue = queue.SimpleQueue()


async def set_title(title):
    p1.title.text = title


# TODO convert all of these 3 functions into one
@without_document_lock
async def use_ble_source():
    service_uuid = os.environ["SERVICE_UUID"]
    device_name = os.environ["DEVICE_NAME"]
    doc.add_next_tick_callback(partial(set_title, title="Scanning..."))
    source = None
    while source is None:
        print(f"Scanning for source with name {device_name}")
        devices = await BleakScanner.discover()
        for d in devices:
            print(d)
        selected_device = [d for d in devices if device_name in str(d.name)]
        device_uuid = selected_device[0].address
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
            data: DeviceData = data_queue.get_nowait()
            with open(filename, "a") as f:
                print(
                    data.timestamp,
                    *data.data_xyz,
                    data.button_state,
                    sep=",",
                    file=f,
                )
            counter += 1
        except queue.Empty:
            print(f"Queue empty, read {counter} items")
            return


@without_document_lock
async def read_data_from_source(source: Source):
    previous_button_state = 0.0
    previous_predicted_state = 0.0
    async for data in source.read_data():
        data_queue.put(data)
        print(
            f"Timestamp: {data.timestamp}, accelerometer data: {data.data_xyz}, "
            f"button state: {data.button_state}, model decision: {data.model_prediction}, "
            f"model score: {data.model_score}"
        )

        if data.button_state and data.button_state != previous_button_state:
            doc.add_next_tick_callback(
                partial(
                    button_clicks.update,
                    text=f"clicks: {int(button_clicks.text.split(': ')[1]) + 1}",
                )
            )
        previous_button_state = data.button_state

        if data.model_prediction and data.model_prediction != previous_predicted_state:
            doc.add_next_tick_callback(
                partial(
                    model_predictions.update,
                    text=f"predictions: {int(model_predictions.text.split(': ')[1]) + 1}",
                )
            )
        previous_predicted_state = data.model_prediction

        doc.add_next_tick_callback(
            partial(
                update_plot,
                x=data.data_xyz[0],
                y=data.data_xyz[1],
                z=data.data_xyz[2],
                button_state=data.button_state,
                button_pred_score=data.model_score,
                button_pred=data.model_prediction,
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

doc.add_root(row(column(row(p1, p2), button), button_clicks, model_predictions))
doc.add_next_tick_callback(use_dummy_source)
