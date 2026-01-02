import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
from ...immutables import ProjectPaths

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1 (list or tuple): [xmin, ymin, xmax, ymax] for the first box.
        box2 (list or tuple): [xmin, ymin, xmax, ymax] for the second box.
        
    Returns:
        float: The IoU value, between 0 and 1.
    """
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the union
    union_area = float(box1_area + box2_area - intersection_area)

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def parse_annotations(file_path):
    """
    Parses the annotation file and groups bounding boxes by image ID.
    
    Args:
        file_path (str): Path to the annotations.txt file.
        
    Returns:
        dict: A dictionary mapping image_id to a list of bounding boxes.
    """
    image_annotations = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            image_id = parts[0]
            # The class label parts[1] is ignored for this analysis
            
            box_data = parts[2:]
            boxes = []
            for box_str in box_data:
                try:
                    coords = [int(c) for c in box_str.split(',')[:4]]
                    # Ensure coords are valid (xmin < xmax, ymin < ymax)
                    if coords[0] < coords[2] and coords[1] < coords[3]:
                        boxes.append(coords)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse box string '{box_str}' in line for {image_id}")
            
            # Use the image_id as the key to store all boxes found on that line
            image_annotations[image_id].extend(boxes)
            
    return image_annotations

def main(annotation_file=ProjectPaths.det_annotations_org):
    """
    Main function to run the analysis.
    """
    print(f"--- Starting Analysis of {annotation_file} ---")
    
    annotations = parse_annotations(annotation_file)
    all_ious = []

    for image_id, boxes in sorted(annotations.items()):
        num_findings = len(boxes)
        print(f"\n--- Image: {image_id} ---")
        print(f"Number of findings: {num_findings}")

        if num_findings < 2:
            print("Not enough findings to calculate IoU.")
            continue

        image_ious = []
        # Use itertools.combinations to get all unique pairs of boxes
        for box1, box2 in itertools.combinations(boxes, 2):
            iou = calculate_iou(box1, box2)
            image_ious.append(iou)
            # You can uncomment the line below for extremely detailed output
            # print(f"  - IoU between {box1} and {box2}: {iou:.4f}")

        all_ious.extend(image_ious)
        
        min_iou = min(image_ious)
        max_iou = max(image_ious)
        avg_iou = sum(image_ious) / len(image_ious)
        
        print(f"IoU Stats for this image:")
        print(f"  - Minimum IoU: {min_iou:.4f}")
        print(f"  - Maximum IoU: {max_iou:.4f}")
        print(f"  - Average IoU: {avg_iou:.4f}")

    print("\n\n--- Overall Dataset IoU Analysis ---")
    
    if not all_ious:
        print("No IoU values were calculated. Ensure your annotations file has images with multiple boxes.")
        return

    all_ious = np.array(all_ious)
    
    print(f"Total pairwise IoUs calculated: {len(all_ious)}")
    print(f"Overall Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Overall Median IoU: {np.median(all_ious):.4f}")
    print(f"Overall Std Dev IoU: {np.std(all_ious):.4f}")
    
    # Calculate percentiles to understand the distribution
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = np.percentile(all_ious, percentiles)
    for p, v in zip(percentiles, percentile_values):
        print(f"{p}th percentile: {v:.4f}")

    # Generate and save a histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_ious, bins=50, range=(0, 1), edgecolor='black')
    plt.title('Distribution of Pairwise IoU in Ground Truth Annotations')
    plt.xlabel('Intersection over Union (IoU)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    histogram_path = '/iou_distribution.jpg'
    plt.savefig(ProjectPaths.visualize_det_datasets + histogram_path)
    print(f"\nHistogram of IoU distribution saved to '{histogram_path}'")


if __name__ == '__main__':
    main()