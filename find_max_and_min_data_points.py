import os

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)

def find_min_max_datapoints(root_dir):
    min_datapoints = float('inf')
    max_datapoints = 0
    min_file = ''
    max_file = ''

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                datapoints = count_lines(file_path)
                
                if datapoints < min_datapoints:
                    min_datapoints = datapoints
                    min_file = file_path
                
                if datapoints > max_datapoints:
                    max_datapoints = datapoints
                    max_file = file_path

    return min_datapoints, min_file, max_datapoints, max_file

# Set the root directory
root_directory = "Bluetooth Datasets/Bluetooth Datasets/Dataset 250 Msps"

# Find min and max datapoints
min_dp, min_file, max_dp, max_file = find_min_max_datapoints(root_directory)

print(f"Minimum datapoints: {min_dp} in file: {min_file}")
print(f"Maximum datapoints: {max_dp} in file: {max_file}")
