import pandas as pd
from ...immutables import ProjectPaths

def get_max_birads(birads_string):
    """
    Parses a BIRADS string (e.g., '4', '3$2') and returns the highest BIRADS score as an integer.
    Handles non-string inputs and parsing errors.
    """
    if not isinstance(birads_string, str):
        return 0  # Return a neutral value for non-string or NaN inputs

    try:
        # Split by '$' and convert to integer, then find the max
        scores = [int(score) for score in birads_string.split('$')]
        return max(scores)
    except (ValueError, TypeError):
        # Handle cases where conversion to int fails
        return 0

def is_consistent(birads_score, pathology_label):
    """
    Checks if the BIRADS score is consistent with the pathology label.

    Args:
        birads_score (int): The maximum BIRADS score for the image.
        pathology_label (str): The pathology classification ('Benign', 'Normal', 'Malignant').

    Returns:
        bool: True if consistent, False otherwise.
    """
    benign_labels = ['Benign', 'Normal']
    malignant_labels = ['Malignant']

    # Rule: BI-RADS 1, 2, 3 should correspond to Benign or Normal labels
    if birads_score in [1, 2, 3]:
        return pathology_label in benign_labels
    
    # Rule: BI-RADS 4, 5, 6 should correspond to Malignant label
    elif birads_score in [4, 5, 6]:
        return pathology_label in malignant_labels
        
    # Any other case (e.g., BIRADS 0 or parsing error) is treated as inconsistent for this filter
    return False

def filter_and_harmonize_annotations(annotations_file_path, segmentations_file_path):
    """
    Filters and harmonizes an annotations CSV file.
    1. Promotes 'Normal' to 'Benign' within a patient/side group if a benign
       finding exists and a segmentation mask is present for any image in that group.
    2. Removes patient/side groups with conflicting 'Benign'/'Malignant' labels.
    3. Removes patient/side groups with inconsistent BI-RADS and pathology labels.
    """
    try:
        df = pd.read_csv(annotations_file_path)
        df_seg = pd.read_csv(segmentations_file_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # Create a set of image names that have segmentation masks for efficient lookup
    # df_seg['#filename'] contains the image filenames with .jpg extension. df['Image_name'] have no extension.
    images_with_masks = set(df_seg['#filename'].str.replace('.jpg', '', regex=False).unique())

    # Extract Patient ID and Side for grouping
    df['Patient_ID'] = df['Image_name'].apply(lambda x: x.split('_')[0])
    df['Side'] = df['Image_name'].apply(lambda x: x.split('_')[1])

    groups_to_remove = set()
    rows_to_update_to_benign = []

    # --- First Pass: Harmonize labels and find irreconcilable groups ---
    print("--- Pass 1: Harmonizing labels and identifying conflicting diagnoses ---")
    
    grouped = df.groupby(['Patient_ID', 'Side'])

    for name, group in grouped:
        unique_labels = set(group['Pathology Classification/ Follow up'].unique())

        # Rule: Exclude groups with conflicting 'Benign'/'Normal' and 'Malignant' diagnoses
        has_malignant = 'Malignant' in unique_labels
        has_benign_or_normal = bool(unique_labels.intersection({'Benign', 'Normal'}))
        
        if has_malignant and has_benign_or_normal:
            print(f"CONFLICT: Group {name} has both Malignant and Benign/Normal labels. Marking for removal.")
            groups_to_remove.add(name)
            continue

        # Rule: Promote 'Normal' to 'Benign' if they coexist in a group AND a mask is present
        if 'Benign' in unique_labels and 'Normal' in unique_labels:
            # Check if any image in this group has a segmentation mask
            if not set(group['Image_name']).isdisjoint(images_with_masks):
                normal_indices = group[group['Pathology Classification/ Follow up'] == 'Normal'].index
                rows_to_update_to_benign.extend(normal_indices)
                print(f"HARMONIZE: Group {name} contains 'Normal' and 'Benign' with a mask. Promoting {len(normal_indices)} 'Normal' labels to 'Benign'.")

    # Apply the promotions in a single, efficient step
    if rows_to_update_to_benign:
        df.loc[rows_to_update_to_benign, 'Pathology Classification/ Follow up'] = 'Benign'
        print(f"\nApplied a total of {len(rows_to_update_to_benign)} label promotions ('Normal' -> 'Benign').\n")

    # --- Second Pass: Check BI-RADS consistency on the (potentially modified) DataFrame ---
    print("--- Pass 2: Checking for BI-RADS and pathology inconsistencies ---")
    
    birads_inconsistent_groups = set()
    for index, row in df.iterrows():
        # Skip check if group is already marked for removal from the first pass
        if (row['Patient_ID'], row['Side']) in groups_to_remove:
            continue

        max_birads = get_max_birads(row['BIRADS'])
        pathology = row['Pathology Classification/ Follow up']

        if not is_consistent(max_birads, pathology):
            birads_inconsistent_groups.add((row['Patient_ID'], row['Side']))

    # Combine all groups that need to be removed
    all_groups_to_remove = groups_to_remove.union(birads_inconsistent_groups)

    print(f"\nFound {len(groups_to_remove)} patient/side groups with conflicting diagnoses.")
    if birads_inconsistent_groups:
        print(f"Found {len(birads_inconsistent_groups)} additional patient/side groups with BI-RADS inconsistencies.")
    print(f"Total unique groups to be removed: {len(all_groups_to_remove)}.")
    
    if all_groups_to_remove:
        print("Groups to be removed:", sorted(list(all_groups_to_remove)))

    # --- Final Step: Filter the DataFrame and save results ---
    
    mask_to_remove = df.apply(lambda row: (row['Patient_ID'], row['Side']) in all_groups_to_remove, axis=1)

    df_filtered = df[~mask_to_remove]
    df_removed = df[mask_to_remove]

    # Save the filtered data to new CSVs
    filtered_output_path = ProjectPaths.dual_stream_det + '/classification_data_preparation/annotations_consistent_harmonized.csv'
    removed_output_path = ProjectPaths.dual_stream_det + '/classification_data_preparation/annotations_removed_conflicts.csv'

    df_filtered.to_csv(filtered_output_path, index=False)
    df_removed.to_csv(removed_output_path, index=False)

    print(f"\nFiltering complete.")
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows kept: {len(df_filtered)}")
    print(f"Number of rows removed: {len(df_removed)}")
    print(f"\nConsistent, harmonized data saved to '{filtered_output_path}'")
    print(f"Removed inconsistent data saved to '{removed_output_path}' for review.")


if __name__ == "__main__":
    # Paths to the input CSV files
    ANNOTATIONS_FILE = ProjectPaths.annotations_all_sheet_modified
    SEGMENTATIONS_FILE = ProjectPaths.segmentations
    
    # Run the filtering and harmonization process
    filter_and_harmonize_annotations(ANNOTATIONS_FILE, SEGMENTATIONS_FILE)