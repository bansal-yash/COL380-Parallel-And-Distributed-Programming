import os
import re
import csv

def parse_perf_file(file_path):
    """Parse a single perf data file to extract relevant percentages."""
    with open(file_path, 'r') as file:
        content = file.read()

    metrics = {
        'matrix_multiply': 0.0,
        'read_matrix': 0.0,
        'write_matrix': 0.0
    }

    # Regular expressions for each function
    patterns = {
        'matrix_multiply': r'(\d+\.\d+)%\s+main\s+main\s+\[\.\]\s+matrixMultiply.*',
        'read_matrix': r'(\d+\.\d+)%\s+main\s+main\s+\[\.\]\s+readMatrix',
        'write_matrix': r'(\d+\.\d+)%\s+main\s+main\s+\[\.\]\s+writeMatrix'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics

def process_perf_files(input_dir, output_csv):
    """Process all matching perf data files and write data to a CSV."""
    data = []

    # Iterate over sizes and types
    for size in range(1000, 6000, 1000):
        for type_ in range(6):
            file_name = f"perf_data_{size}_{type_}.txt"
            file_path = os.path.join(input_dir, file_name)

            if os.path.exists(file_path):
                metrics = parse_perf_file(file_path)
                metrics['file_name'] = file_name  # Add file name for reference
                data.append(metrics)
            else:
                print(f"File not found: {file_path}")

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'matrix_multiply', 'read_matrix', 'write_matrix']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print(f"Data written to {output_csv}")

# Specify input directory and output CSV file
input_directory = ''
output_csv_file = 'perf_record_data.csv'

process_perf_files(input_directory, output_csv_file)
