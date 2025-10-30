
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from ..utils import cvtColor
from ..immutables import ProjectPaths
from ..medical_image_utils import min_max_normalise
class DualStreamDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes # Bbox class (just "mass")
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip().split()
        # ASSUMED FORMAT: Image_Name img_label box1 box2...
        # EXAMPLE FORMAT: P12_L_CM_MLO 1 980,1574,1152,1746,0 669,1238,769,1338,0
        # img_label is 0 for Benign, 1 for Malignant.
        # Box format is min_x,min_y,max_x,max_y,class_id. class_id is always 0 for "mass".
       
        image_name = str(line[0])

        if "CM" in image_name:
            # ASSUMPTION: DM(LE) image path is derived by replacing a substring.
            ce_image_path = ProjectPaths.det_dataset + "/" + image_name
            le_image_path = ProjectPaths.det_dataset + "/" + image_name.replace('CM', 'DM')
        else:
            ce_image_path = ProjectPaths.det_dataset + "/" + image_name.replace('DM', 'CM')
            le_image_path = ProjectPaths.det_dataset + "/" + image_name
        
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[2:]])
        
        le_image = Image.open(le_image_path)
        ce_image = Image.open(ce_image_path)
        
        # NOTE: For simplicity, only resize is implemented. For full augmentations,
        # geometric transforms (resize, crop, flip) must be identical for both
        # images, while color transforms can be separate.
        le_image, ce_image, boxes = self.process_data(le_image, ce_image, boxes, self.input_shape)

        # le_image = np.transpose(preprocess_input(np.array(le_image, dtype=np.float32)), (2, 0, 1))
        # ce_image = np.transpose(preprocess_input(np.array(ce_image, dtype=np.float32)), (2, 0, 1))
        
        le_image = np.transpose(min_max_normalise(np.array(le_image, dtype=np.float32)), (2, 0, 1))
        ce_image = np.transpose(min_max_normalise(np.array(ce_image, dtype=np.float32)), (2, 0, 1))

        return le_image, ce_image, boxes

    def process_data(self, le_image, ce_image, box, input_shape):
        le_image = cvtColor(le_image)
        ce_image = cvtColor(ce_image)
        
        iw, ih = le_image.size
        h, w = input_shape

        # Simple resize
        le_image = le_image.resize((w, h), Image.BICUBIC)
        ce_image = ce_image.resize((w, h), Image.BICUBIC)
        le_image_data = np.array(le_image, np.float32)
        ce_image_data = np.array(ce_image, np.float32)

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

        return le_image_data, ce_image_data, box

def dual_stream_collate(batch):
    le_images, ce_images, bboxes= [], [], []
    for le_img, ce_img, box in batch:
        le_images.append(le_img)
        ce_images.append(ce_img)
        bboxes.append(box)
    le_images = np.array(le_images)
    ce_images = np.array(ce_images)
    return le_images, ce_images, bboxes