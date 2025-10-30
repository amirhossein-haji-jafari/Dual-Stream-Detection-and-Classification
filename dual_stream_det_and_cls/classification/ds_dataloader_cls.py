import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from ..utils import cvtColor
from ..immutables import ProjectPaths
from ..medical_image_utils import min_max_normalise
class DualStreamDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, return_names=False):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.return_names = return_names

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip().split()
        # ASSUMED FORMAT: Image_Name img_label 
        # EXAMPLE FORMAT: P12_L_CM_MLO 1
        # img_label is 0 for Benign, 1 for Malignant and 2 for Normal (if included).
       
        image_name = str(line[0])

        if "CM" in image_name:
            # ASSUMPTION: DM(LE) image path is derived by replacing a substring.
            ce_image_path = ProjectPaths.cls_dataset + "/" + image_name
            le_image_path = ProjectPaths.cls_dataset + "/" + image_name.replace('CM', 'DM')
        else:
            ce_image_path = ProjectPaths.cls_dataset + "/" + image_name.replace('DM', 'CM')
            le_image_path = ProjectPaths.cls_dataset + "/" + image_name
        
        image_label = int(line[1]) 
        
        le_image = Image.open(le_image_path)
        ce_image = Image.open(ce_image_path)
        
        # NOTE: For simplicity, only resize is implemented. For full augmentations,
        # geometric transforms (resize, crop, flip) must be identical for both
        # images, while color transforms can be separate.
        le_image, ce_image = self.process_data(le_image, ce_image, self.input_shape)

        le_image = np.transpose(min_max_normalise(np.array(le_image, dtype=np.float32)), (2, 0, 1))
        ce_image = np.transpose(min_max_normalise(np.array(ce_image, dtype=np.float32)), (2, 0, 1))
        
        if self.return_names:
            return le_image[0:1,:,:], ce_image[0:1,:,:], image_label, image_name
        else:
            return le_image[0:1,:,:], ce_image[0:1,:,:], image_label


    def process_data(self, le_image, ce_image, input_shape):
        le_image = cvtColor(le_image)
        ce_image = cvtColor(ce_image)
        
        h, w = input_shape

        # Simple resize
        le_image = le_image.resize((w, h), Image.BICUBIC)
        ce_image = ce_image.resize((w, h), Image.BICUBIC)
        le_image_data = np.array(le_image, np.float32)
        ce_image_data = np.array(ce_image, np.float32)

        return le_image_data, ce_image_data

def dual_stream_collate(batch):
    if not batch:
        return np.array([]), np.array([]), np.array([])

    if len(batch[0]) == 4:
        le_images, ce_images, labels, image_names = [], [], [], []
        for le_img, ce_img, label, img_name in batch:
            le_images.append(le_img)
            ce_images.append(ce_img)
            labels.append(label)
            image_names.append(img_name)
        le_images = np.array(le_images)
        ce_images = np.array(ce_images)
        labels = np.array(labels)
        return le_images, ce_images, labels, image_names
    else:
        le_images, ce_images, labels = [], [], []
        for le_img, ce_img, label in batch:
            le_images.append(le_img)
            ce_images.append(ce_img)
            labels.append(label)
        le_images = np.array(le_images)
        ce_images = np.array(ce_images)
        labels = np.array(labels)
        return le_images, ce_images, labels