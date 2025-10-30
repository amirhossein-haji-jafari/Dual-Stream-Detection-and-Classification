"""
AP@50: 63.57% ± 5.72%
F1: 61.20% ± 5.00%
Recall: 57.67% ± 9.84%
Precision: 66.53% ± 3.84%
"""

import numpy as np

results = {
    "fold 0":{
        "AP@50": 0.5537,
        "F1": 0.57,
        "Recall": 0.4848,
        "Precision": 0.6957
    },
    "fold 1":{
        "AP@50": 0.6636,
        "F1": 0.63,
        "Recall": 0.6406,
        "Precision": 0.6165
    },
    "fold 2":{
        "AP@50": 0.6922,
        "F1": 0.66,
        "Recall": 0.6947,
        "Precision": 0.6286
    },
    "fold 3":{
        "AP@50": 0.6706,
        "F1": 0.66,
        "Recall": 0.60,
        "Precision": 0.725
    },
    "fold 4":{
        "AP@50": 0.63,
        "F1": 0.60,
        "Recall": 0.58,
        "Precision": 0.6304
    },
}

# calculate mean results with standard deviation
metrics = ["AP@50", "F1", "Recall", "Precision"]
means = {}
stds = {}

for metric in metrics:
    values = [results[fold][metric] for fold in results]
    means[metric] = np.mean(values)
    stds[metric] = np.std(values)

for metric in metrics:
    print(f"{metric}: {means[metric]*100:.2f}% ± {stds[metric]*100:.2f}%")

# test: calculate F1 from Recall and Precision and check if it matches the existing F1
print("\nF1 consistency check (calculated vs existing):")
for fold in results:
    recall = results[fold]["Recall"]
    precision = results[fold]["Precision"]
    f1_calc = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_existing = results[fold]["F1"]
    print(f"{fold}: F1 (calculated) = {f1_calc:.4f}, F1 (existing) = {f1_existing:.4f}, Diff = {abs(f1_calc - f1_existing):.4e}")