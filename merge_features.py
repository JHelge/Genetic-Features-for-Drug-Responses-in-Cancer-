


import csv
import os

def merge_csv_files(folder_path, output_file):
    merged_data = []

    for i in range(1003, 2501):
        file_path = os.path.join(folder_path, f"Drug{i}_analysis/best/features_0.csv")
        if os.path.isfile(file_path):
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if row not in merged_data:
                        merged_data.append(row)

    with open(output_file, 'w', newline='') as output_csv:
        csv_writer = csv.writer(output_csv)
        for row in merged_data:
            csv_writer.writerow(row)

    print("CSV files merged successfully!")
    print(f"Number of rows in merged file: {len(merged_data)}")

folder_path = './'
output_file = 'merged_data.csv'

merge_csv_files(folder_path, output_file)


