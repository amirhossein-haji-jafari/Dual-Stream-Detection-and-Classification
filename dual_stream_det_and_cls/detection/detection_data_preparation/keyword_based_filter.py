"""
remove images from training set based on keywords in report/opinion
"""

from .detection_dataset_definitions import mass_union_mass_enhancement
import pandas as pd
from immutables import ProjectPaths

KEYWORDS = ["foci", "focus", "stippled", "adenosis"]
def main():
    medical_reports = pd.read_csv(ProjectPaths.parsed_reports)
    kept_images = []
    removed_images = []
    for image_name in mass_union_mass_enhancement:
        cm_image_name = ""
        dm_image_name = ""
        if "_CM_" in image_name:
            cm_image_name = image_name
            dm_image_name = image_name.replace("_CM_", "_DM_")
        elif "_DM_" in image_name:
            dm_image_name = image_name
            cm_image_name = image_name.replace("_DM_", "_CM_")
        
        # get report and opinion for the image
        filtered_row = medical_reports[medical_reports['Image_name'] == cm_image_name]
        if not filtered_row.empty:
            report_cm = filtered_row.iloc[0]["report"]
            opinion_cm = filtered_row.iloc[0]["opinion"]
        filtered_row = medical_reports[medical_reports['Image_name'] == dm_image_name]
        if not filtered_row.empty:
            report_dm = filtered_row.iloc[0]["report"]
            opinion_dm = filtered_row.iloc[0]["opinion"]
        

        report = str(report_cm) + " " + str(report_dm)
        report = report.lower()
        opinion = str(opinion_cm) + " " + str(opinion_dm)
        opinion = opinion.lower()
        image_has_keywords = any(keyword in report for keyword in KEYWORDS) or any(keyword in opinion for keyword in KEYWORDS)
        if not image_has_keywords:
            kept_images.append(image_name)
        else:
            print(f"Removing image {image_name} due to keywords in report/opinion.")
            removed_images.append(image_name)



    print("len mass_u_enhancement_u_nme:", len(mass_union_mass_enhancement))
    print(f"Kept images ({len(kept_images)}):\n", kept_images)





if __name__ == "__main__":
    main()
