import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_and_plot(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        data = []
        for line in lines:
            if line[0] != "#":
                data.append(line.split())

    times = []

    powers = []

    for line in lines[1:]:

        if "Step Information" not in line or "time" not in line:
            try:
                time = float(line.split()[0])
                times.append(time)

                power = float(line.split()[1])
                powers.append(power)
            except:
                pass

    print("Average Power: ")
    print(np.mean(powers))


file_path = "neuron.txt"
analyze_and_plot(file_path)
