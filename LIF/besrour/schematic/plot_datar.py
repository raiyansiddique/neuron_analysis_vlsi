import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot(file_path):
    # Initialize lists to store the final data
    caps, isyns, vdds, freqs, joules_per_spike = [], [], [], [], []
    current_vouts, current_times, current_powers = [], [], []

    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#") or line.strip() == "":
                continue

            if "Step Information" in line:
                cap_val, isyn_val, vdd_val = process_step_information(line)
                if current_vouts and current_times:  # Check if there's data to process
                    spiking_frequency, average_power = calculate_metrics(current_times, current_vouts, current_powers)
                    freqs.append(spiking_frequency)
                    joules_per_spike.append(average_power)

                # Reset for next set of measurements
                current_vouts, current_times, current_powers = [], [], []
                # Append new step information
                caps.append(cap_val)
                isyns.append(isyn_val)
                vdds.append(vdd_val)
            else:
                process_line_data(line, current_vouts, current_times, current_powers)

        # Process the last set of data
        if current_vouts and current_times:
            spiking_frequency, average_power = calculate_metrics(current_times, current_vouts, current_powers)
            freqs.append(spiking_frequency)
            joules_per_spike.append(average_power)

    # Plot the results
    plot_data(caps, isyns, vdds, freqs, joules_per_spike)

def process_step_information(line):
    parts = line.split()
    cap_val = parse_value(parts[2])
    isyn_val = parse_value(parts[3])
    vdd_val = parse_value(parts[4])
    return cap_val, isyn_val, vdd_val

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

def calculate_metrics(times, vouts, powers):
    frequency, (start_index, end_index) = get_spike_frequency(times, vouts)
    average_power = np.mean(powers[start_index:end_index]) if start_index < end_index else 0
    return frequency, average_power / frequency if frequency > 0 else 0

def parse_value(s):
    multiplier = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3}
    number_part = "".join(filter(str.isdigit, s)) or '0'
    unit = s[-1]
    return float(number_part) * multiplier.get(unit, 1)

def get_spike_frequency(times, vouts):
    threshold = 0.1
    spike_count = 0
    start_time = -1
    end_time = -1

    for i in range(1, len(vouts)):
        if vouts[i] > threshold and vouts[i - 1] <= threshold:
            spike_count += 1
            if start_time == -1:
                start_time = times[i]
            end_time = times[i]

    frequency = spike_count / (end_time - start_time) if start_time != -1 and end_time != start_time else 0
    return frequency, (0, len(times) - 1)  # Return entire range if no spike detected

def plot_data(caps, isyns, vdds, freqs, joules_per_spike):
    # Create a DataFrame from the collected data
    data = pd.DataFrame({
        'Capacitance': caps,
        'ISyn': isyns,
        'Vdd': vdds,
        'Frequency': freqs,
        'EnergyPerSpike': joules_per_spike
    })

    # Remove entries with Capacitance <= 0 if any
    data = data[data['Capacitance'] > 0]

    # Unique Vdd values for coloring
    unique_vdds = data['Vdd'].unique()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

    # Unique Vdd values for coloring
    colors_vdd = plt.cm.jet(np.linspace(0, 1, len(unique_vdds)))

    # Plot Maximum Frequency vs. Capacitance for different Vdds on the first subplot
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data['Vdd'] == vdd]
        max_freq_per_cap = data_filtered.groupby('Capacitance')['Frequency'].max().reset_index()
        axs[0].plot(max_freq_per_cap['Capacitance'], max_freq_per_cap['Frequency'], marker='o', linestyle='-', color=colors_vdd[i], label=f'Vdd={vdd}')
    axs[0].set_xlabel('Capacitance (F)')
    axs[0].set_ylabel('Maximum Frequency (Hz)')
    axs[0].set_title('Maximum Frequency vs. Capacitance for different Vdds')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Energy Per Spike at Max Frequency vs. Capacitance for different Vdds on the second subplot
    for i, vdd in enumerate(unique_vdds):
        data_filtered = data[data['Vdd'] == vdd]
        energy_at_max_freq = data_filtered.groupby('Capacitance').apply(lambda x: x[x['Frequency'] == x['Frequency'].max()]['EnergyPerSpike'].iloc[0]).reset_index(name='EnergyPerSpike')
        axs[1].plot(energy_at_max_freq['Capacitance'], energy_at_max_freq['EnergyPerSpike'], marker='o', linestyle='-', color=colors_vdd[i], label=f'Vdd={vdd}')
    axs[1].set_xlabel('Capacitance (F)')
    axs[1].set_ylabel('Energy Per Spike at Max Frequency (J)')
    axs[1].set_title('Energy Per Spike at Max Frequency vs. Capacitance for different Vdds')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
    plt.show()

    # Plot Spiking Frequency vs. Synaptic Current for different Capacitances
    unique_caps = sorted(data['Capacitance'].unique())
    colors_cap = plt.cm.viridis(np.linspace(0, 1, len(unique_caps)))
    plt.figure(figsize=(12, 6))
    for i, cap in enumerate(unique_caps):
        data_filtered = data[data['Capacitance'] == cap]
        sorted_data = data_filtered.sort_values(by='ISyn')
        plt.plot(sorted_data['ISyn'], sorted_data['Frequency'], marker='o', linestyle='-', color=colors_cap[i], label=f'Cap={cap}F')
    plt.xlabel('Synaptic Current (ISyn) [A]')
    plt.ylabel('Spiking Frequency [Hz]')
    plt.title('Spiking Frequency vs. Synaptic Current for different Capacitances')
    plt.legend(title='Capacitance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3D Scatter Plot for Neuron Frequency Response
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(data['EnergyPerSpike'].min(), data['EnergyPerSpike'].max())
    colors = plt.cm.viridis(norm(data['EnergyPerSpike']))
    sc = ax.scatter(data['ISyn'], data['Capacitance'], data['Frequency'], c=colors, marker="o")
    ax.set_xlabel("Synaptic Current (A)")
    ax.set_ylabel("Capacitance (F)")
    ax.set_zlabel("Spiking Frequency")
    ax.set_title("Neuron Frequency Response")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Energy Per Spike (J)")
    plt.show()

file_path = "50f_10u.txt"
analyze_and_plot(file_path)
