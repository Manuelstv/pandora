import json
import heapq
import statistics
import numpy as np

# Load the JSON file
with open('datasets/360INDOOR/annotations/instances_val2017_transformed.json') as f:
    data = json.load(f)

# Ensure data is a list of annotations
annotations = data if isinstance(data, list) else data.get('annotations', [])

# Initialize variables to store the kappa values and their locations
kappa_values = []
min_annotation = None
max_annotation = None

# Iterate through all annotations
for annotation in annotations:
    if isinstance(annotation, dict):
        bbox = annotation.get('bbox', [])
        if len(bbox) >= 4:
            kappa_values.append(bbox[3])
            if min_annotation is None or bbox[3] < min_annotation['kappa']:
                min_annotation = {'kappa': bbox[3], 'annotation': annotation}
            if max_annotation is None or bbox[3] > max_annotation['kappa']:
                max_annotation = {'kappa': bbox[3], 'annotation': annotation}

# Compute statistics
min_kappa = min(kappa_values)
max_kappa = max(kappa_values)
mean_kappa = statistics.mean(kappa_values)
median_kappa = statistics.median(kappa_values)
std_dev_kappa = statistics.stdev(kappa_values)
variance_kappa = statistics.variance(kappa_values)
range_kappa = max_kappa - min_kappa
top_smallest_kappas = heapq.nsmallest(5, kappa_values)
top_biggest_kappas = heapq.nlargest(5, kappa_values)

# Compute percentiles
percentiles = [25, 50, 75, 90, 95, 99]
percentile_values = {p: np.percentile(kappa_values, p) for p in percentiles}

print(f"The minimum value of kappa is: {min_kappa}")
print(f"The maximum value of kappa is: {max_kappa}")
print(f"The mean value of kappa is: {mean_kappa}")
print(f"The median value of kappa is: {median_kappa}")
print(f"The standard deviation of kappa is: {std_dev_kappa}")
print(f"The variance of kappa is: {variance_kappa}")
print(f"The range of kappa is: {range_kappa}")
print(f"The top 5 smallest values of kappa are: {top_smallest_kappas}")
print(f"The top 5 biggest values of kappa are: {top_biggest_kappas}")

for p, value in percentile_values.items():
    print(f"The {p}th percentile of kappa is: {value}")

if min_annotation:
    print(f"The minimum kappa value is found in the annotation: {min_annotation['annotation']}")
if max_annotation:
    print(f"The maximum kappa value is found in the annotation: {max_annotation['annotation']}")