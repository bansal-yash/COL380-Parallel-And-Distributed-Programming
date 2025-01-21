import os
import re
import csv

def parse_performance_file(file_path,report_file_name,prof_file_name,size,type):
    data = {
        "Size": size,
        "Type": type,
        "Python_Time": 0,
        "Perf_Time": 0,
        "User_time": 0,
        "Sys_time": 0,
        "Cache_References": 0,
        "Cache_Misses": 0,
        "Cache_Miss_Rate": 0,
        "Cache_Hits": 0,
        "Cache_Hit_Rate": 0,
        "Cycles": 0,
        "Instructions": 0,
        "Instructions_per_cycle": 0,
        "Percent_Matrix_Multiply_perf": 0,
        "User_time_Matrix_Multiply_perf": 0,
        "Percent_Read_matrix_perf": 0,
        "Percent_Write_matrix_perf": 0,
        "Percent_Matrix_Multiply_gprof": 0,
        "User_time_Matrix_Multiply_gprof": 0,
        "Percent_Read_matrix_gprof": 0,
        "Percent_Write_matrix_gprof": 0,
    }
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data['Size'] = size
        data['Type'] = type
        for line in lines:
            if 'Time Duration' in line:
                data['Python_Time'] = float(line.split()[2])
            if 'seconds time elapsed' in line:
                data['Perf_Time'] = float(line.split()[0])
            elif 'seconds user' in line:
                data['User_time'] = float(line.split()[0])
            elif 'seconds sys' in line:
                data['Sys_time'] = float(line.split()[0])
            elif 'cache-references' in line:
                data['Cache_References'] = int(line.split()[0].replace(',', ''))
            elif 'cache-misses' in line:
                data['Cache_Misses'] = int(line.split()[0].replace(',', ''))
                data['Cache_Miss_Rate'] = float(line.split("#")[1].split('%')[0])
                data['Cache_Hits'] = data['Cache_References'] - data['Cache_Misses']
                data['Cache_Hit_Rate'] = 100-data['Cache_Miss_Rate']
            elif 'cycles' in line:
                data['Cycles'] = int(line.split()[0].replace(',', ''))
            elif 'instructions' in line:
                data['Instructions'] = int(line.split()[0].replace(',', ''))
                data['Instructions_per_cycle'] = float(line.split('#')[1].split()[0])
    with open(report_file_name,"r") as file:
        lines = file.readlines()
        for line in lines:

            if 'matrixMultiply' in line:
                data['Percent_Matrix_Multiply_perf'] = float(line.split('%')[0])
                data['User_time_Matrix_Multiply_perf'] = data['Percent_Matrix_Multiply_perf']*0.01*data['User_time']
            elif 'readMatrix' in line:
                data['Percent_Read_matrix_perf'] = float(line.split('%')[0])
            elif 'writeMatrix' in line:
                data['Percent_Write_matrix_perf'] = float(line.split('%')[0])
    with open(prof_file_name,"r") as file:
        lines = file.readlines()
        for line in lines:
            if 'Copyright' in line:
                break
            if 'matrixMultiply' in line:
                print(line.split())
                data['Percent_Matrix_Multiply_gprof'] = float(line.split()[0])
                data['User_time_Matrix_Multiply_gprof'] = float(line.split()[2])
            elif 'readMatrix' in line:
                data['Percent_Read_matrix_gprof'] = float(line.split()[0])
            elif 'writeMatrix' in line:
                data['Percent_Write_matrix_gprof'] = float(line.split()[0])      
    return data

def process_files( output_csv):
    all_data = []
    
    # Iterate through all files in the folder
    for size in [1000,2000,3000,4000,5000]:
        for type in range(6):
            file_name = f"out_{size}_{type}.txt"
            report_file_name = f"perf_data_{size}_{type}.txt"
            prof_file_name = f"prof_out_{size}_{type}.txt"
            file_data = parse_performance_file(file_name,report_file_name,prof_file_name,size,type)
            all_data.append(file_data)
    
    # Write data to CSV
    if all_data:
        with open(output_csv, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Data written to {output_csv}")
    else:
        print("No valid data found to write.")

if __name__ == "__main__":
    output_csv = "performance_summary.csv"
    process_files(output_csv)