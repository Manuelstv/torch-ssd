import os
import json
import random

# Define the directory where the JSON files are located
json_directory = "/home/mstveras/360-obj-det/annotations"

# Initialize an empty set to store unique class names
unique_class_names = set()

# Loop through each JSON file in the directory
for json_file in os.listdir(json_directory):
    if json_file.endswith(".json"):
        # Construct the full path to the JSON file
        json_path = os.path.join(json_directory, json_file)
        
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the class names from the "boxes" field
        for box in data.get("boxes", []):
            class_name = box[-1]  # The class name is the last element in each "box" list
            unique_class_names.add(class_name)

# Convert the set to a sorted list
unique_class_names_list = sorted(list(unique_class_names))

# Generate random colors for each class
distinct_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(unique_class_names_list))]

# Create a dictionary to associate each class name with a color
class_color_dict = dict(zip(unique_class_names_list, distinct_colors))

# Print the class names and their associated colors
print("Class Names:", tuple(unique_class_names_list))
print("Distinct Colors:", distinct_colors)
print("Class-Color Dictionary:", class_color_dict)
