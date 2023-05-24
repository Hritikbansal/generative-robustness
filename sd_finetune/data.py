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



class ImageCaptionDataset(Dataset):

    def __init__(self, filename, tokenizer, image_transform, caption_key = 'caption', image_key = 'image'):

        df = pd.read_csv(filename)
        
        self.transform = image_transform
        self.captions  = df[caption_key].tolist()
        self.tokenizer = tokenizer
        self.images    = df[image_key].tolist()

        self.t_captions= tokenizer(self.captions, max_length = tokenizer.model_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        item = {}
        item['image_location'] = self.images[index]
        item['caption'] = self.captions[index]
        item['input_ids']  = self.t_captions['input_ids'][index]
        item['pixel_values'] = self.transform(Image.open(self.images[index]).convert('RGB'))

        return item