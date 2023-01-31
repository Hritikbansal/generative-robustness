import os
import csv
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from .data import PromptDataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionImg2ImgPipeline

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--split", type = str, default = "train", help = "Path to eval test data")
parser.add_argument("--dataset", type = str, default = "imagenet", help = "dataset name")
parser.add_argument("--data_dir", type = str, default = "/home/data/ImageNet1K/validation", help = "Path to eval test data")
parser.add_argument("--save_image_gen", type = str, default = None, help = "Path saved generated images")
parser.add_argument("--save_real", type = str, default = None, help = "save real data")
parser.add_argument('--diversity', action='store_true', help='diverse captions or not')

args = parser.parse_args()
accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok = True)

def preprocess(image):
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def generate_images(pipe, dataloader, args):
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe = pipe.to(accelerator.device)
    filename  = os.path.join(args.save_image_gen, 'i2i.csv')
    with torch.no_grad():
        for original_images, image_locations, captions, labels in tqdm(dataloader):
            indices = list(filter(lambda x: not os.path.exists(os.path.join(args.save_image_gen, image_locations[x])), range(len(image_locations))))
            if len(indices) == 0:
                continue
            original_images, image_locations, captions, labels = map(lambda x: [x[i] for i in indices], (original_images, image_locations, captions, labels))
            original_images = torch.stack(original_images).to(accelerator.device)
            images = pipe(prompt = captions, image = original_images, strength = 1).images
            for index in range(len(images)):
                os.makedirs(os.path.join(args.save_image_gen, os.path.dirname(image_locations[index])), exist_ok = True)
                path = os.path.join(args.save_image_gen, image_locations[index])
                images[index].save(path)
                with open(filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile) 
                    csvwriter.writerow([path, labels[index], captions[index]])
                if args.save_real:
                    real_img_path = os.path.join(args.save_image_gen, 'real')
                    os.makedirs(real_img_path, exist_ok = True)
                    shutil.copy(os.path.join(args.data_dir, image_locations[index]), real_img_path)


def main():
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    dataset    = PromptDataset(args.data_dir, split = args.split, dataset = args.dataset, diversity = args.diversity, i2i = True, transform = preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
    generate_images(pipe, dataloader, args)

if __name__ == "__main__":
    main()