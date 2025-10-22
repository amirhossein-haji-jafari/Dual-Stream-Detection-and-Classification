# Save this file as single_stream_dataloader.py

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from ..utils import cvtColor, preprocess_input
from ..immutables import ProjectPaths
from ..medical_image_utils import min_max_normalise

class SingleStreamDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, include_stream='dm'):
        super().__init__()
        
        self.include_stream = str(include_stream).lower()  # 'both', 'cm', or 'dm'
        
        # For each line, create the appropriate entries depending on include_stream.
        self.expanded_annotation_lines = []
        for line in annotation_lines:
            parts = line.strip().split()
            if not parts:
                continue
            image_name = parts[0]
            partner_name = image_name.replace("_CM_", "_DM_") if "_CM_" in image_name else image_name.replace("_DM_", "_CM_")
            partner_line = f"{partner_name} {' '.join(parts[1:])}"

            if self.include_stream == 'both':
                # keep both original and partner
                self.expanded_annotation_lines.append(line)
                self.expanded_annotation_lines.append(partner_line)
            elif self.include_stream == 'cm':
                # ensure CM variant is present
                if "_CM_" in image_name:
                    self.expanded_annotation_lines.append(line)
                else:
                    self.expanded_annotation_lines.append(partner_line.replace("  ", " "))
            elif self.include_stream == 'dm':
                # ensure DM variant is present
                if "_DM_" in image_name:
                    self.expanded_annotation_lines.append(line)
                else:
                    self.expanded_annotation_lines.append(partner_line.replace("  ", " "))
            else:
                # fallback to both if unknown mode passed
                self.expanded_annotation_lines.append(line)
                self.expanded_annotation_lines.append(partner_line)

        self.length = len(self.expanded_annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.expanded_annotation_lines[index].strip().split()
        
        image_name = str(line[0])
        image_path = ProjectPaths.det_dataset + "/" + image_name
        
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[2:]])
        
        image = Image.open(image_path)
        
        image, boxes = self.process_data(image, boxes, self.input_shape)

        # image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        image = np.transpose(min_max_normalise(np.array(image, dtype=np.float32)), (2, 0, 1))
        
        return image, boxes

    def process_data(self, image, box, input_shape):
        image = cvtColor(image)
        
        iw, ih = image.size
        h, w = input_shape

        # Simple resize
        image = image.resize((w, h), Image.BICUBIC)
        image_data = np.array(image, np.float32)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * w / iw
            box[:, [1, 3]] = box[:, [1, 3]] * h / ih
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

def single_stream_collate(batch):
    images, bboxes= [], []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes