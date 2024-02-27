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
    vdds = []

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

            vdd_str = line.split()[4]
            vdd_val = float("".join(filter(str.isdigit, vdd_str)))
            if vdd_str[-1] == "f":
                vdd_val = vdd_val * 1e-15
            elif vdd_str[-1] == "p":
                vdd_val = vdd_val * 1e-12
            elif vdd_str[-1] == "n":
                vdd_val = vdd_val * 1e-9
            elif vdd_str[-1] == "�":
                vdd_val = vdd_val * 1e-6
            elif vdd_str[-1] == "m":
                vdd_val = vdd_val * 1e-3

            vdds.append(vdd_val)

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
        ########### DO WHATEVER WITH vdds ###########
    data = pd.DataFrame({
        'Capacitance': caps,
        'Frequency': freqs,
        'EnergyPerSpike': joules_per_spike,
        'ISyn': isyns,
        'Vdd': vdds
    })

    # Group by Capacitance to find the maximum frequency and corresponding energy per spike for each capacitance
    max_freq_per_cap = data.groupby('Capacitance')['Frequency'].max().reset_index()
    energy_at_max_freq = data.groupby('Capacitance').apply(lambda x: x[x['Frequency'] == x['Frequency'].max()]['EnergyPerSpike'].iloc[0]).reset_index(name='EnergyPerSpike')

    unique_vdds = data['Vdd'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_vdds)))  # Color map for different Vdds
    data = data[data['Capacitance'] > 0]
    plt.figure(figsize=(10, 5))
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data['Vdd'] == vdd]
        max_freq_per_cap = data_filtered.groupby('Capacitance')['Frequency'].max().reset_index()
        plt.plot(max_freq_per_cap['Capacitance'], max_freq_per_cap['Frequency'], marker='o', linestyle='-', color=colors[i], label=f'Vdd={vdd}')
    plt.xlabel('Capacitance (F)')
    plt.ylabel('Maximum Frequency (Hz)')
    plt.title('Maximum Frequency vs. Capacitance for different Vdds')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data['Vdd'] == vdd]
        # For energy, we still need to compute or filter data similarly
        energy_at_max_freq = data_filtered.groupby('Capacitance').apply(lambda x: x[x['Frequency'] == x['Frequency'].max()]['EnergyPerSpike'].iloc[0]).reset_index(name='EnergyPerSpike')
        plt.plot(energy_at_max_freq['Capacitance'], energy_at_max_freq['EnergyPerSpike'], marker='o', linestyle='-', color=colors[i], label=f'Vdd={vdd}')
    plt.xlabel('Capacitance (F)')
    plt.ylabel('Energy Per Spike at Max Frequency (J)')
    plt.title('Energy Per Spike at Max Frequency vs. Capacitance for different Vdds')
    plt.legend()
    plt.grid(True)
    plt.show()

    unique_caps = sorted(data['Capacitance'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_caps)))  # Color map for different capacitances

    plt.figure(figsize=(12, 6))

    # Plot lines for each capacitance
    for i, cap in enumerate(unique_caps):
        data_filtered = data[data['Capacitance'] == cap]
        sorted_data = data_filtered.sort_values(by='ISyn')  # Sort by ISyn for smooth lines
        plt.plot(sorted_data['ISyn'], sorted_data['Frequency'], marker='o', linestyle='-', color=colors[i], label=f'Cap={cap}F')

    plt.xlabel('Synaptic Current (ISyn) [A]')
    plt.ylabel('Spiking Frequency [Hz]')
    plt.title('Spiking Frequency vs. Synaptic Current for different Capacitances')
    plt.legend(title='Capacitance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(min(joules_per_spike), max(joules_per_spike))
    colors = plt.cm.viridis(norm(joules_per_spike))

    ax.scatter(isyns, caps, freqs, c=colors, marker="o")

    ax.set_xlabel("Synaptic Current (A)")
    ax.set_ylabel("Capacitance (F)")
    ax.set_zlabel("Spiking Frequency")

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


file_path = "neuron_5-9V_10u.txt"
analyze_and_plot(file_path)
