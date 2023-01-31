import os
import PIL
import glob
import torch
import torchvision
import requests
import random
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image, ImageFile
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader


ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

class CondLDM(Dataset):

    def __init__(self, num_images_per_class = 1300):
        
        im100_folders = ['n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406', 'n02091831', 'n04517823', 'n04589890', 'n03062245', 'n01773797', 'n01735189', 'n07831146', 'n07753275', 'n03085013', 'n04485082', 'n02105505', 'n01983481', 'n02788148', 'n03530642', 'n04435653', 'n02086910', 'n02859443', 'n13040303', 'n03594734', 'n02085620', 'n02099849', 'n01558993', 'n04493381', 'n02109047', 'n04111531', 'n02877765', 'n04429376', 'n02009229', 'n01978455', 'n02106550', 'n01820546', 'n01692333', 'n07714571', 'n02974003', 'n02114855', 'n03785016', 'n03764736', 'n03775546', 'n02087046', 'n07836838', 'n04099969', 'n04592741', 'n03891251', 'n02701002', 'n03379051', 'n02259212', 'n07715103', 'n03947888', 'n04026417', 'n02326432', 'n03637318', 'n01980166', 'n02113799', 'n02086240', 'n03903868', 'n02483362', 'n04127249', 'n02089973', 'n03017168', 'n02093428', 'n02804414', 'n02396427', 'n04418357', 'n02172182', 'n01729322', 'n02113978', 'n03787032', 'n02089867', 'n02119022', 'n03777754', 'n04238763', 'n02231487', 'n03032252', 'n02138441', 'n02104029', 'n03837869', 'n03494278', 'n04136333', 'n03794056', 'n03492542', 'n02018207', 'n04067472', 'n03930630', 'n03584829', 'n02123045', 'n04229816', 'n02100583', 'n03642806', 'n04336792', 'n03259280', 'n02116738', 'n02108089', 'n03424325', 'n01855672', 'n02090622']
        im100_classes = [452, 64, 374, 236, 993, 176, 882, 904, 503, 74, 57, 959, 953, 508, 872, 228, 122, 421, 599, 858, 157, 449, 994, 608, 151, 209, 15, 876, 246, 766, 455, 857, 131, 119, 234, 90, 45, 936, 479, 272, 665, 653, 659, 158, 960, 765, 908, 703, 407, 560, 317, 938, 724, 748, 331, 619, 120, 267, 155, 708, 368, 772, 167, 494, 180, 431, 342, 854, 305, 54, 268, 667, 166, 277, 662, 798, 313, 498, 299, 222, 682, 593, 775, 674, 592, 137, 758, 717, 606, 281, 796, 211, 620, 830, 544, 275, 242, 570, 99, 169]

        self.folders = im100_folders * num_images_per_class
        self.classes = im100_classes * num_images_per_class

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        return idx, self.classes[idx], self.folders[idx]

class PromptDataset(Dataset):

    def __init__(self, root, split = 'train', dataset = 'imagenet', diversity = True, i2i = False, transform = None):

        self.root  = root
        self.split = split
        config = eval(open(os.path.join(self.root, 'classes.py'), 'r').read())
        self.diversity = diversity
        self.templates = config["templates"] if self.diversity else lambda x: f'a photo of a {x}'
        self.folders   = os.listdir(os.path.join(self.root, self.split))
        if dataset == 'imagenet':
            df = pd.read_csv(os.path.join(self.root, 'folder_to_class.csv'))
            df = df[['folder', 'class']]
            self.folder_to_class = df.set_index('folder').T.to_dict('list')
        elif dataset == 'cifar10': 
            # folder names in cifar are class names
            self.folder_to_class = {k:k for k in self.folders}

        self.images = []
        self.classes = []
        for folder in self.folders:
            class_images = os.listdir(os.path.join(self.root, split, folder))
            class_images = list(map(lambda x: os.path.join(split, folder, x), class_images))
            self.images  = self.images + class_images
            self.classes = self.classes + ([self.folder_to_class[folder]] * len(class_images))
        
        self.i2i = i2i
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.diversity:
            index = random.randint(0, len(self.templates) - 1)
            caption = self.templates[index](self.classes[idx][0])
        else:
            caption = self.templates(self.classes[idx][0])
        
        image_location = self.images[idx]
        
        if self.i2i:
            image = Image.open(os.path.join(self.root, image_location)).convert("RGB")
            return self.transform(image), image_location, caption, self.classes[idx][0]
        else:
            return image_location, caption, self.classes[idx][0]


class ImageDataset(Dataset):
    def __init__(self, root, transform, split = 'train'):

        self.root = root
        self.transform = transform
        config = eval(open(os.path.join(self.root, 'classes.py'), 'r').read())
        self.templates = config["templates"]
        self.folders = os.listdir(os.path.join(self.root, split))
        self.images = []
        for folder in self.folders:
            class_images = os.listdir(os.path.join(self.root, split, folder))
            class_images = list(map(lambda x: os.path.join(split, folder, x), class_images))
            self.images  = self.images + class_images

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        return self.images[idx], image