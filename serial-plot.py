# this program plots output of serial port live
import time

import serial
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow

# set safe font for macos
plt.rcParams["font.family"] = "arial"


def makeFig(data):
    """Make plot of accelerometer data.
    data: list of tuples of accelerometer data for 3 coordinates
    """
    data = np.array(data)
    plt.plot(data[:, 0], data[:, 1], "ro-", label="X")
    plt.plot(data[:, 0], data[:, 2], "go-", label="Y")
    plt.plot(data[:, 0], data[:, 3], "bo-", label="Z")
    plt.legend(loc="upper left")
    plt.title("Accelerometer Data")
    plt.grid(True)
    plt.ylim(-4, 4)
    plt.ylabel("Acceleration (G)")
    plt.xlabel("Sample Number")


def main():
    # global variables
    data = []
    arduinoData = None
    while True:
        try:
            arduinoData = serial.Serial("/dev/cu.usbmodem1413101", 9600)  # serial port
            break
        except OSError as e:
            print(f"Could not connect to serial port {e}")
            print("Trying again in 5 seconds...")
            time.sleep(5)
            continue
    plt.ion()  # turn interactive mode on
    # main loop
    while True:
        while arduinoData.inWaiting() == 0:  # wait for data
            pass
        arduinoString = arduinoData.readline()  # read data
        dataArray = arduinoString.split(b"\t")  # split data
        values = tuple(map(float, dataArray))  # convert to float
        data.append(values)  # add to array
        drawnow(lambda: makeFig(data))  # draw plot
        plt.pause(0.000001)  # pause plot
        if len(data) > 50:  # check array size
            data.pop(0)  # remove first value


if __name__ == "__main__":
    main()
