import pandas as pd

def calculate_average_power_consumption(file_path):
    # Load the data from the file
    data = pd.read_csv(file_path, delim_whitespace=True)

    # Correctly reference the first column as time
    time_column_name = data.columns[0]
    
    # Calculate time intervals between each measurement
    time_intervals = data[time_column_name].diff().fillna(0)
    
    # Multiply each power value by its subsequent time interval
    # Assuming the time intervals are in the first column, shift(-1) to use the next interval for each measurement
    instantaneous_power = data.iloc[:, 1:].multiply(time_intervals, axis=0)
    
    # Calculate the total time covered by the dataset
    total_time = data[time_column_name].iloc[-1] - data[time_column_name].iloc[0]
    print(total_time)
    # Calculate the weighted average power consumption for each component
    weighted_average_power_consumption = instantaneous_power.sum() / total_time
    
    # Sum the average power consumption of all components
    total_average_power_consumption = weighted_average_power_consumption.sum()
    
    # Return the total average power consumption
    return total_average_power_consumption

# Example usage:
file_path = 'power_low.txt'  # Replace this with the path to your file
total_avg_power = calculate_average_power_consumption(file_path)
print(f"Total Average Power Consumption: {total_avg_power} Watts")
