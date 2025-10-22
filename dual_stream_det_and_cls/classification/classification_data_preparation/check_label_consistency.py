import pandas as pd
import os
from ...immutables import ProjectPaths

# Define the output file for the report
OUTPUT_REPORT_FILE = os.path.join(ProjectPaths.dual_stream_cls, "mismatched_labels_report.txt")

def find_label_mismatches(annotations_file_path: str, output_file_path: str):
    """
    Finds and reports inconsistencies in image labels within a dataset.

    The script assumes that all images for a specific patient and breast side
    (e.g., 'P315_R') should have the same 'Pathology Classification/ Follow up' label.

    Args:
        annotations_file_path (str): The path to the CSV annotation file.
        output_file_path (str): The path to save the detailed report of mismatches.
    """
    print(f"Reading annotations from: {annotations_file_path}")
    
    try:
        # Load the dataset from the provided CSV file path
        df = pd.read_csv(annotations_file_path)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at '{annotations_file_path}'. Please check the path in `immutables.py`.")
        return

    # --- Step 1: Extract grouping keys (Patient ID and Side) from the image name ---
    # Example 'Image_name': P315_R_DM_CC
    # 'Patient_ID': P315
    # 'Side': R
    try:
        df['Patient_ID'] = df['Image_name'].apply(lambda x: x.split('_')[0])
        df['Side'] = df['Image_name'].apply(lambda x: x.split('_')[1])
    except IndexError:
        print("Error: Could not parse 'Patient_ID' and 'Side' from 'Image_name'.")
        print("Please ensure image names follow the 'PatientID_Side_Type_View' format.")
        return

    # --- Step 2: Group the DataFrame by Patient and Side ---
    grouped = df.groupby(['Patient_ID', 'Side'])

    mismatched_groups = []
    
    # --- Step 3: Iterate through each group and check for label consistency ---
    for name, group in grouped:
        # Get the unique labels within the current group
        label_column = 'Pathology Classification/ Follow up'
        unique_labels = group[label_column].unique()

        # If there is more than one unique label, it's a mismatch
        if len(unique_labels) > 1:
            mismatched_groups.append({
                "patient_id": name[0],
                "side": name[1],
                "labels": list(unique_labels),
                "details": group[['Image_name', label_column]].to_dict('records')
            })

    # --- Step 4: Report the findings ---
    if not mismatched_groups:
        print("\n--- All Clear! ---")
        print("No label mismatches found across all patient/side groups.")
        # Create an empty report file to indicate the check was run
        with open(output_file_path, 'w') as f:
            f.write("Label consistency check completed.\n")
            f.write("No mismatches were found.\n")
        return

    print("\n--- Mismatched Labels Found! ---")
    print(f"Found {len(mismatched_groups)} groups with inconsistent labels.")
    print(f"A detailed report is being saved to: {output_file_path}\n")

    # Open the output file to write the report
    with open(output_file_path, 'w') as f:
        f.write("--- Report on Mismatched Image Labels ---\n\n")
        
        for mismatch in mismatched_groups:
            # Format the output for both console and file
            header = f"Group: Patient {mismatch['patient_id']}, Side {mismatch['side']}"
            conflicts = f"Conflicting Labels: {mismatch['labels']}"
            
            # Print to console
            print("-" * 40)
            print(header)
            print(conflicts)
            print("Images in this group:")
            
            # Write to file
            f.write("-" * 40 + "\n")
            f.write(header + "\n")
            f.write(conflicts + "\n")
            f.write("Images in this group:\n")

            for record in mismatch['details']:
                line = f"  - {record['Image_name']}: {record[label_column]}"
                print(line)
                f.write(line + "\n")
            
            print("-" * 40 + "\n")
            f.write("-" * 40 + "\n\n")

    print(f"Report successfully saved.")


if __name__ == "__main__":
    # The path to the annotations CSV is managed by the ProjectPaths class
    annotations_path = ProjectPaths.annotations_all_sheet_modified
    
    # Run the function to find and report mismatches
    find_label_mismatches(annotations_path, OUTPUT_REPORT_FILE)