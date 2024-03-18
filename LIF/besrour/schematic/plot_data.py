import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def analyze_and_plot(file_path):
    # Initialize lists to store the final data
    caps, isyns, vdds, freqs, joules_per_spike = [], [], [], [], []
    current_vouts, current_times, current_powers = [], [], []

    with open(file_path, "r", encoding="unicode_escape") as file:
        for line in file:
            if line.startswith("#") or line.strip() == "":
                continue

            if "Step Information" in line:
                isyn_val, cap_val, vdd_val = process_step_information(line)
                if current_vouts and current_times:  # Check if there's data to process
                    spiking_frequency, average_power = calculate_metrics(
                        current_times, current_vouts, current_powers
                    )
                    freqs.append(spiking_frequency)
                    joules_per_spike.append(average_power)

                # Reset for next set of measurements
                current_vouts, current_times, current_powers = [], [], []
                # Append new step information
                caps.append(cap_val)
                isyns.append(isyn_val)
                vdds.append(round(vdd_val, 5))
            else:
                process_line_data(line, current_vouts, current_times, current_powers)

        # Process the last set of data
        if current_vouts and current_times:
            spiking_frequency, average_power = calculate_metrics(
                current_times, current_vouts, current_powers
            )
            freqs.append(spiking_frequency)
            joules_per_spike.append(average_power)

    # Plot the results
    plot_data(caps, isyns, vdds, freqs, joules_per_spike)


def process_step_information(line):
    parts = line.split()
    isyn_val = parse_value(parts[3])
    cap_val = parse_value(parts[4])
    vdd_val = parse_value(parts[2])
    return isyn_val, cap_val, vdd_val


def process_line_data(line, current_vouts, current_times, current_powers):
    parts = line.split()
    if len(parts) >= 2:
        try:
            time = float(parts[0])
            vout = float(parts[-1])
            power = sum([float(i) for i in parts[1:-1]])
            current_times.append(time)
            current_vouts.append(vout)
            current_powers.append(power)
        except ValueError:
            pass  # Ignore lines that cannot be processed


def parse_value(s):
    if "." in s:
        multiplier = {
            "f": 1e-15,
            "p": 1e-12,
            "n": 1e-9,
            "�": 1e-6,
            "µ": 1e-6,
            "m": 1e-3,
            "k": 1e3,
            # "default": 1e-6,
        }
        decimal_place = 0
        for char in s[-1:0:-1]:
            if char == ".":
                break
            decimal_place += 1

        number_part = "".join(filter(str.isdigit, s)) or "0"
        number_part = (
            number_part[0 : -decimal_place + 1]
            + "."
            + number_part[-decimal_place + 1 :]
        )
        unit = s[-1]
        return float(number_part) * multiplier.get(unit, 1)
    multiplier = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "µ": 1e-6,
        "�": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        # "default": 1e-6,
    }
    number_part = "".join(filter(str.isdigit, s)) or "0"
    unit = s[-1]
    return float(number_part) * multiplier.get(unit, 1)


def calculate_metrics(times, vouts, powers):
    # frequency, (start_index, end_index) = get_spike_frequency(times, vouts)
    # average_power = (
    #     np.mean(powers[start_index:end_index]) if start_index < end_index else 0
    # )
    # return frequency, average_power / frequency if frequency > 0 else 0

    spike_count, (start_index, end_index) = get_spike_count(vouts)
    total_energy = 0
    for i in range(start_index, end_index):
        current_time = times[i]
        try:
            next_time = times[i + 1]
            time_diff = next_time - current_time
        except:
            break
        total_energy += powers[i] * time_diff

    if spike_count == 0 or (times[end_index] - times[start_index]) == 0:
        return 0, 0

    frequency = spike_count / (times[end_index] - times[start_index])
    joules_per_spike = total_energy / spike_count
    return frequency, joules_per_spike


def get_spike_frequency(times, vouts):
    threshold = 0.1
    spike_count = 0
    start_time_idx = -1
    end_time_idx = -1
    start_time = times[0]
    end_time = times[-1]

    for i in range(1, len(vouts)):
        if vouts[i] > threshold and vouts[i - 1] <= threshold:
            spike_count += 1
            if start_time_idx == -1:
                start_time = times[i]
                start_time_idx = i
            end_time = times[i]
            end_time_idx = i

    if start_time_idx != -1 and end_time_idx != start_time_idx:

        frequency = spike_count / (end_time - start_time)

    else:
        frequency = 0
    return frequency, (
        start_time_idx,
        end_time_idx,
    )  # Return entire range if no spike detected


def get_spike_count(vouts):
    threshold = 0.1
    spike_count = 0
    start_time_idx = -1
    end_time_idx = -1

    for i in range(1, len(vouts)):
        if vouts[i] > threshold and vouts[i - 1] <= threshold:
            spike_count += 1
            if start_time_idx == -1:
                start_time_idx = i
            end_time_idx = i

    return spike_count, (start_time_idx, end_time_idx)


def plot_data(caps, isyns, vdds, freqs, joules_per_spike):
    # Create a DataFrame from the collected data
    data = pd.DataFrame(
        {
            "Capacitance": caps,
            "ISyn": isyns,
            "Vdd": vdds,
            "Frequency": freqs,
            "EnergyPerSpike": joules_per_spike,
        }
    )

    # Remove entries with Capacitance <= 0 if any
    data = data[data["Capacitance"] > 0]
    print(data["Capacitance"])
    unique_vdds = data["Vdd"].unique()

    # Increase default font sizes and line widths using plt.rc
    plt.rc("font", size=20)  # Increase font size
    plt.rc(
        "axes", titlesize=20, labelsize=18
    )  # Increase axes title and label font sizes
    plt.rc("xtick", labelsize=18)  # Increase x-axis tick label font size
    plt.rc("ytick", labelsize=18)  # Increase y-axis tick label font size
    plt.rc(
        "legend", fontsize=18
    )  # Increase legend font sizeplt.rcParams['font.weight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    line_width = 4  # Set line width

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

    # Assuming 'unique_vdds' is defined and 'data' is your DataFrame

    # Unique Vdd values for coloring
    colors_vdd = plt.cm.jet(np.linspace(0, 1, len(unique_vdds)))

    # Plot Maximum Frequency vs. Capacitance for different Vdds on the first subplot
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data["Vdd"] == vdd]
        max_freq_per_cap = (
            data_filtered.groupby("Capacitance")["Frequency"].max().reset_index()
        )
        axs[0].plot(
            max_freq_per_cap["Capacitance"],
            max_freq_per_cap["Frequency"],
            marker="o",
            linestyle="-",
            color=colors_vdd[i],
            linewidth=line_width,
            markersize=10,
            label=f"VDD={vdd} V",
        )
    axs[0].set_xlabel("Capacitance (fF)", fontweight="bold")
    axs[0].set_ylabel("Maximum Frequency (MHz)", fontweight="bold")
    axs[0].set_title(
        "Maximum Frequency vs. Capacitance for different VDDs", fontweight="bold"
    )
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # Adjusting x-axis labels to display in nF
    axs[0].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x*10**15)}")
    )

    # Adjusting y-axis labels to display in MHz
    axs[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, pos: f"{int(y*10**-6)}")
    )

    plt.tight_layout()

    # Plot Energy Per Spike at Max Frequency vs. Capacitance for different Vdds on the second subplot
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data["Vdd"] == vdd]
        energy_at_max_freq = (
            data_filtered.groupby("Capacitance")
            .apply(
                lambda x: x[x["Frequency"] == x["Frequency"].max()][
                    "EnergyPerSpike"
                ].iloc[0]
            )
            .reset_index(name="EnergyPerSpike")
        )
        axs[1].plot(
            energy_at_max_freq["Capacitance"],
            energy_at_max_freq["EnergyPerSpike"],
            marker="o",
            linestyle="-",
            color=colors_vdd[i],
            linewidth=line_width,
            markersize=10,
            label=f"VDD={vdd} V",
        )
    axs[1].set_xlabel("Capacitance (fF)", fontweight="bold")
    axs[1].set_ylabel("Energy Per Spike\nat Max Frequency (fJ)", fontweight="bold")
    axs[1].set_title(
        "Energy Per Spike at Max Frequency vs. Capacitance\nfor different VDDs",
        fontweight="bold",
    )
    axs[1].legend(loc="lower right")
    axs[1].grid(True)

    # Adjusting x-axis labels to display in nF
    axs[1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x*10**15)}")
    )

    # Adjusting y-axis labels to display in MHz
    axs[1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, pos: f"{int(y*10**15)}")
    )

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
    plt.show()

    # # Plot Spiking Frequency vs. Synaptic Current for different Capacitances
    # unique_caps = sorted(data["Capacitance"].unique())
    # colors_cap = plt.cm.viridis(np.linspace(0, 1, len(unique_caps)))
    # plt.figure(figsize=(12, 6))
    # for i, cap in enumerate(unique_caps):
    #     data_filtered = data[data["Capacitance"] == cap]
    #     sorted_data = data_filtered.sort_values(by="ISyn")
    #     plt.plot(
    #         sorted_data["ISyn"],
    #         sorted_data["Frequency"],
    #         marker="o",
    #         linestyle="-",
    #         color=colors_cap[i],
    #         label=f"Cap={cap}F",
    #     )
    # plt.xlabel("Synaptic Current (ISyn) [A]")
    # plt.ylabel("Spiking Frequency [Hz]")
    # plt.title("Spiking Frequency vs. Synaptic Current for different Capacitances")
    # plt.legend(title="Capacitance", bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    while True:

        # Assuming 'data' is your pandas DataFrameq
        print("VDD:")
        vdd_thing = float(input("vdd: "))
        # vdd_thing = 0.9
        tolerance = 1e-9
        filtered_data = data[np.isclose(data["Vdd"], vdd_thing, atol=tolerance)]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        print("Energy Per Spike:")
        print(filtered_data["EnergyPerSpike"].max())
        energy = int(input("Divisor: "))

        print("+++++")
        print(filtered_data["EnergyPerSpike"].max())

        # Normalize color based on energy per spike d`irectly
        norm = Normalize(
            vmin=filtered_data["EnergyPerSpike"].min(),
            vmax=filtered_data["EnergyPerSpike"].max() / energy,
            # vmax=filtered_data["EnergyPerSpike"].max(),
        )
        cmap = cm.viridis

        # Apply the normalization and colormap directly to the EnergyPerSpike values for coloring
        colors = cmap(norm(filtered_data["EnergyPerSpike"]))

        # Scatter plot
        sc = ax.scatter(
            filtered_data["ISyn"],
            filtered_data["Capacitance"],
            filtered_data["Frequency"],
            c=colors,
            marker="o",
        )

        # Set labels with units
        ax.set_xlabel("Synaptic Current (nA)")
        ax.set_ylabel("Capacitance (fF)")
        ax.set_zlabel("Spiking Frequency (MHz)")
        ax.set_title("Neuron Frequency Response")

        # Apply formatting for tick labels
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x*10**9)}")
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, pos: f"{int(y*10**15)}")
        )
        ax.zaxis.set_major_formatter(
            ticker.FuncFormatter(lambda z, pos: f"{int(z*10**-6)}")
        )

        # Color bar indicating energy per spike, with actual values displayed
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        # Update color bar to reflect energy in femtojoules
        cbar.set_label("Energy Per Spike (fJ)")
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x*10**15)}")
        )

        plt.show()


# file_path = "50f_10u.txt"
file_path = "neuron_22_fin_1.txt"
analyze_and_plot(file_path)
