import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot(file_path):
    # Read the file
    data = pd.read_csv(file_path, sep='\t', comment='#', header=None, encoding='ISO-8859-1')
    data.columns = ['time', 'V(vout)', 'I(I1)']

    # Convert columns to numeric and filter out non-numeric rows
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['V(vout)'] = pd.to_numeric(data['V(vout)'], errors='coerce')
    data['I(I1)'] = pd.to_numeric(data['I(I1)'], errors='coerce')
    data.dropna(subset=['time', 'V(vout)', 'I(I1)'], inplace=True)

    # Define the spike threshold
    threshold = 0.4

    # Identify unique current levels
    unique_current_levels = data['I(I1)'].unique()

    # Initialize lists for spike counts and durations
    spike_counts = []
    durations = []

    # Process each current level
    for current_level in unique_current_levels:
        # Extract data for the current level
        current_data = data[data['I(I1)'] == current_level]

        # Calculate spikes
        voltage = current_data['V(vout)']
        spike_starts = (voltage > threshold) & (voltage.shift(1) <= threshold)
        spike_counts.append(spike_starts.sum())

        # Calculate duration
        duration = current_data['time'].iloc[-1] - current_data['time'].iloc[0]
        durations.append(duration)

    # Calculate spiking frequencies in kHz
    spiking_frequencies_kHz = [(count / duration) / 1000 for count, duration in zip(spike_counts, durations)]

    # Find the maximum frequency
    max_frequency_kHz = max(spiking_frequencies_kHz)
    max_frequency_index = spiking_frequencies_kHz.index(max_frequency_kHz)
    max_frequency_current_level = unique_current_levels[max_frequency_index]

    # Printing the maximum frequency
    print(f"Maximum Frequency: {max_frequency_kHz:.2f} kHz at Current Level: {max_frequency_current_level * 1e9:.2f} nA")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(unique_current_levels * 1e9, spiking_frequencies_kHz, marker='o')  # Current in nA
    plt.xlabel('Current Level (nA)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Spiking Frequency (kHz) at Each Current Level')
    plt.grid(True)
    plt.show()

# Replace 'path_to_your_file.txt' with the path to your file
file_path = 'neuron.txt'
analyze_and_plot(file_path)
