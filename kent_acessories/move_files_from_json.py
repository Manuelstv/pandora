import json
import shutil
import os

# Path to the JSON file
json_file_path = '/home/mstveras/mmdetection-2.x/datasets/360INDOOR/annotations/instances_val2017.json'

# Directories
source_directory = '/home/mstveras/mmdetection-2.x/data/360INDOOR/images'
destination_directory = '/home/mstveras/mmdetection-2.x/datasets/360INDOOR/images/val2017'

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Read JSON content
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract file names
file_names = [image['file_name'] for image in data['images']]

# Move files
for file_name in file_names:
    #print(file_name)
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)
    
    # Check if the file exists in the source directory before moving
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name}")
    else:
        print(f"File not found: {file_name}")

print("Operation completed.")