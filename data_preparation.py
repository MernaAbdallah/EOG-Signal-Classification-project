import shutil
import os
import csv

# Define source and destination directories
sources = [
    'Train-H_CSV.csv',
    'Train-V_CSV.csv',
    'Test-H_CSV.csv',
    'Test-V_CSV.csv'
]
destinations = [
    'NewTrain-H',
    'NewTrain-V',
    'NewTest-H',
    'NewTest-V'
]

# Loop through each source and destination pair
for source, dest in zip(sources, destinations):
    # Create the destination directory if it doesn't exist
    os.makedirs(dest, exist_ok=True)

    # Copy the source file to the destination folder
    shutil.copy(source, dest)

    # Get the file name without extension
    base_name = os.path.splitext(os.path.basename(source))[0]

    # Write the source file to a CSV file in the destination folder
    output_csv_path = os.path.join(dest, base_name + '_CSV.csv')

    # Create a CSV file with data from the source text file
    with open(source, 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file.readlines()]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([base_name])
        writer.writerow(data)

    print(f"Conversion to CSV completed for {source}. Combined data saved to:", output_csv_path)
