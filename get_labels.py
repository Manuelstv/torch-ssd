import json
import os

def extract_labels(annotations_dir):
    labels = set()  # Use a set to automatically eliminate duplicates
    for annotation_file in os.listdir(annotations_dir):
        with open(os.path.join(annotations_dir, annotation_file)) as f:
            annotations = json.load(f)['boxes']
            for annotation in annotations:
                labels.add(annotation[6])  # Assuming the label is the 7th element in each annotation
    return labels

annotations_dir = '/home/mstveras/360-obj-det/annotations'
unique_labels = extract_labels(annotations_dir)

# Print out the labels and a formatted label_map dictionary
print(f'Unique Labels: {unique_labels}')
print('Label Map Dictionary:')
print('{')
for i, label in enumerate(sorted(unique_labels)):
    print(f"    '{label}': {i},")
print('}')

# Output can be copied and pasted into your CustomDataset class

