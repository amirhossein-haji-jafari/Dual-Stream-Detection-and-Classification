import pandas as pd
from ..immutables import ProjectPaths

ann = pd.read_csv (ProjectPaths.annotations_all_sheet_modified)

labels_field = 'Pathology Classification/ Follow up'
count_fields = ['Benign', 'Malignant']

# count Benign and Malignant labels in ann file
counts = {field: 0 for field in count_fields}
for label in ann[labels_field]:
    if pd.isna(label):
        continue
    label = label.strip()
    if label in counts:
        counts[label] += 1

print("Label counts:")
for field, count in counts.items():
    print(f"{field}: {count}")