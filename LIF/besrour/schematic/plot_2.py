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

    caps = []
    isyns = []

    vouts = []
    current_vouts = []

    times = []
    current_times = []

    powers = []
    current_powers = []

    for line in lines[1:]:
        vout = 0
        if "Step Information" in line:
            cap_str = line.split()[2]
            cap_val = float("".join(filter(str.isdigit, cap_str)))
            if cap_str[-1] == "f":
                cap_val = cap_val * 1e-15
            elif cap_str[-1] == "p":
                cap_val = cap_val * 1e-12
            elif cap_str[-1] == "n":
                cap_val = cap_val * 1e-9
            elif cap_str[-1] == "�":
                cap_val = cap_val * 1e-6
            elif cap_str[-1] == "m":
                cap_val = cap_val * 1e-3

            caps.append(cap_val)

            isyn_str = line.split()[3]
            isyn_val = float("".join(filter(str.isdigit, isyn_str)))
            if isyn_str[-1] == "f":
                isyn_val = isyn_val * 1e-15
            elif isyn_str[-1] == "p":
                isyn_val = isyn_val * 1e-12
            elif isyn_str[-1] == "n":
                isyn_val = isyn_val * 1e-9
            elif isyn_str[-1] == "�":
                isyn_val = isyn_val * 1e-6
            elif isyn_str[-1] == "m":
                isyn_val = isyn_val * 1e-3

            isyns.append(isyn_val)

            if len(current_vouts) == 0:
                current_vouts = []
            else:
                vouts.append(current_vouts)
                current_vouts = []

            if len(current_times) == 0:
                current_times = []
            else:
                times.append(current_times)
                current_times = []

            if len(current_powers) == 0:
                current_powers = []
            else:
                powers.append(current_powers)
                current_powers = []

            continue

        if "Step Information" not in line or "time" not in line:
            try:
                time = float(line.split()[0])
                current_times.append(time)

                vout = float(line.split()[-1])
                current_vouts.append(vout)

                power = sum([float(i) for i in line.split()[1:-1]])
                current_powers.append(power)
            except:
                pass

    vouts.append(current_vouts)
    times.append(current_times)
    powers.append(current_powers)

    spiking_time_interval_indicies = []

    freqs = []
    for i in range(len(vouts)):
        spiking_frequency, spiking_time_interval_index = get_spike_frequency(
            times[i], vouts[i]
        )
        freqs.append(spiking_frequency)
        spiking_time_interval_indicies.append(spiking_time_interval_index)

    average_powers = []
    for i, power in enumerate(powers):
        print(power)
        average_powers.append(
            np.mean(
                power[
                    spiking_time_interval_indicies[i][
                        0
                    ] : spiking_time_interval_indicies[i][1]
                ]
            )
        )

    joules_per_spike = get_energy_per_spike(average_powers, freqs)

    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection="3d")

    # norm = plt.Normalize(min(joules_per_spike), max(joules_per_spike))
    # colors = plt.cm.viridis(norm(joules_per_spike))

    # ax.scatter(isyns, caps, joules_per_spike, marker="o")

    # ax.set_xlabel("Synaptic Current (A)")
    # ax.set_ylabel("Capacitance (F)")
    # ax.set_zlabel("Joules per Spike (J)")

    # ax.set_title("Neuron Power Consumption")

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(min(joules_per_spike), max(joules_per_spike))
    colors = plt.cm.viridis(norm(joules_per_spike))

    ax.scatter(isyns, caps, freqs, c=colors, marker="o")

    ax.set_xlabel("Synaptic Current (A)")
    ax.set_ylabel("Capacitance (F)")
    ax.set_zlabel("Spiking Frequency (kHz)")

    ax.set_title("Neuron Frequency Response")

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cbar.set_label("Joules per Spike (J)")

    plt.show()


def get_spike_frequency(time, vout):
    threshold = 0.4
    spike_count = 0

    start_time = -1
    end_time = -1

    start_time_index = 0
    end_time_index = -1
    for i in range(1, len(vout)):
        if vout[i] > threshold and vout[i - 1] <= threshold:
            spike_count += 1
            if start_time == -1:
                start_time = time[i]
                start_time_index = i
            end_time = time[i]
            end_time_index = i

    if start_time == -1:
        start_time = 0
    if end_time == -1:
        end_time = 0

    try:
        return spike_count / (end_time - start_time), [start_time_index, end_time_index]
    except ZeroDivisionError:
        return 0, [0, -1]


def get_energy_per_spike(average_powers, freqs):
    joules_per_spike = []

    for average_power, freq in zip(average_powers, freqs):
        if freq == 0:
            joules_per_spike.append(0)
            continue
        joules_per_spike.append(average_power / freq)

    return joules_per_spike


file_path = "neuron.txt"
analyze_and_plot(file_path)
