import os
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
file_path = 'data_full.csv'  # Replace with your CSV file path
output_folder = 'plots'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read CSV file
data = pd.read_csv(file_path)

# Mapping for permutation labels
permutation_labels = ['Type 0:- IJK', 'Type 1:- IKJ', 'Type 2:- JIK', 'Type 3:- JKI', 'Type 4:- KIJ', 'Type 5:- KJI']

# Grouping the data
matrix_dimensions = data['Matrix Dimension'].unique()
permutation_numbers = range(6)  # 0 to 5

# Columns to plot
columns_to_plot = [
    'Total Time Python', 'Total Time Perf', 'Total User Time', 'Total System Time',
    'Total Cache References', 'Total Cache Misses', 'Total Cache Hits',
    'Cache Miss Rate', 'Cache Hit Rate', 'Total Instructions', 'Total Cycles',
    'Instructions per Cycle', 'Percent User Time in Matrix Multiplication - Perf',
    'User Time in Matrix Multiplication - Perf', 'Percent Time in Read Matrix - Perf',
    'Percent time in Write Matrix - Perf', 'Percent Time in Main - Perf',
    'Percent Time in Matrix Multiplication - GProf', 'Time in Matrix Multiplication - Gprof',
    'Percent Time in Read Matrix - Gprof', 'Percent time in Write Matrix - Gprof'
]

# Iterate over each column to plot
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))

    # Plot each permutation
    for perm_number in permutation_numbers:
        subset = data[data['Permutation Number'] == perm_number]
        plt.plot(
            subset['Matrix Dimension'], subset[column], label=permutation_labels[perm_number]
        )

    # Add labels and legend
    plt.title(f'{column} vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, f'{column.replace(" ", "_").replace("-", "_")}.png')
    plt.savefig(plot_path)
    plt.close()

print(f'Plots saved in the folder: {output_folder}')
