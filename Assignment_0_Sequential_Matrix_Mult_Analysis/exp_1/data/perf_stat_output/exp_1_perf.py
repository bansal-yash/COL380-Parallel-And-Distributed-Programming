import os
import re
import csv

def parse_file(file_path):
    """Parse a single file to extract the required performance metrics."""
    with open(file_path, 'r') as file:
        content = file.read()

    metrics = {}

    # Regular expressions for each metric
    patterns = {
        'cache_references': r'(\d[\d,]*)\s+cache-references',
        'cache_misses': r'(\d[\d,]*)\s+cache-misses',
        'cycles': r'(\d[\d,]*)\s+cycles',
        'instructions': r'(\d[\d,]*)\s+instructions',
        'time_elapsed': r'([\d.]+) seconds time elapsed',
        'time_user': r'([\d.]+) seconds user',
        'time_sys': r'([\d.]+) seconds sys',
        'time_duration': r'Time Duration:- ([\d.]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = match.group(1).replace(',', '')  # Remove commas from numbers
        else:
            metrics[key] = None  # If a metric is not found, set it to None

    return metrics

def process_files(input_dir, output_csv):
    """Process all matching files and write data to a CSV."""
    data = []

    # Iterate over sizes and types
    for size in range(1000, 6000, 1000):
        for type_ in range(6):
            file_name = f"out_{size}_{type_}.txt"
            file_path = os.path.join(input_dir, file_name)

            if os.path.exists(file_path):
                metrics = parse_file(file_path)
                metrics['file_name'] = file_name  # Add file name for reference
                data.append(metrics)
            else:
                print(f"File not found: {file_path}")

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'cache_references', 'cache_misses', 'cycles', 'instructions',
                      'time_elapsed', 'time_user', 'time_sys', 'time_duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print(f"Data written to {output_csv}")

# Specify input directory and output CSV file
input_directory = ''
output_csv_file = 'data_exp_1.csv'

process_files(input_directory, output_csv_file)
