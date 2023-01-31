import os
import csv
import torch
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from .data import PromptDataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--split", type = str, default = "train", help = "Path to eval test data")
parser.add_argument("--dataset", type = str, default = "imagenet", help = "dataset name")
parser.add_argument("--data_dir", type = str, default = "/home/data/ImageNet1K/validation", help = "Path to eval test data")
parser.add_argument("--save_image_gen", type = str, default = None, help = "Path saved generated images")
parser.add_argument('--save_real', action='store_true', help='save real or not')
parser.add_argument('--diversity', action='store_true', help='diverse captions or not')

args = parser.parse_args()
accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok = True)

filename  = os.path.join(args.save_image_gen, 'train_captions.csv')
def generate_images(pipe, dataloader, args):
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe = pipe.to(accelerator.device)
    with torch.no_grad():
        for image_locations, captions, labels in tqdm(dataloader):
            indices = list(filter(lambda x: not os.path.exists(os.path.join(args.save_image_gen, image_locations[x])), range(len(image_locations))))
            if len(indices) == 0:
                continue
            image_locations = [image_locations[i] for i in indices]
            captions = [captions[i] for i in indices]
            labels   = [labels[i] for i in indices]
            images   = pipe(captions).images
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
    pipe       = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    dataset    = PromptDataset(args.data_dir, split = args.split, dataset = args.dataset, diversity = args.diversity)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
    generate_images(pipe, dataloader, args)

if __name__ == "__main__":
    main()

'''
accelerate launch --num_cpu_threads_per_process 8 -m generation.generate_images_captions --batch_size 4 --eval_test_data_dir /data0/datasets/ImageNet1K/ILSVRC/Data/CLS-LOC/ --eval_save_image_gen /data0/datasets/ImageNet1K/ILSVRC/Data/ImageVariation/
'''