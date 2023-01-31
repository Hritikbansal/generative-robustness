import os
import csv
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from .data import ImageDataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionImageVariationPipeline

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--split", type = str, default = "train", help = "Path to eval test data")
parser.add_argument("--data_dir", type = str, default = "/home/data/ImageNet1K/validation", help = "Path to eval test data")
parser.add_argument("--save_image_gen", type = str, default = None, help = "Path saved generated images")

args = parser.parse_args()

accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok = True)

def generate_images(pipe, dataloader, args):
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe = pipe.to(accelerator.device)
    filename  = os.path.join(args.save_image_gen, 'images_variation.csv')
    with torch.no_grad():
        for image_locations, original_images in tqdm(dataloader):
            indices = list(filter(lambda x: not os.path.exists(os.path.join(args.save_image_gen, image_locations[x])), range(len(image_locations))))
            if len(indices) == 0:
                continue
            original_images = original_images[indices]
            image_locations = [image_locations[i] for i in indices]
            images = pipe(original_images, guidance_scale = 3).images
            for index in range(len(images)):
                os.makedirs(os.path.join(args.save_image_gen, os.path.dirname(image_locations[index])), exist_ok = True)
                images[index].save(os.path.join(args.save_image_gen, image_locations[index]))

def main():
    model_name_path = "lambdalabs/sd-image-variations-diffusers"
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_name_path, revision = "v2.0")
    
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = ImageDataset(args.data_dir, tform, split = args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
    generate_images(pipe, dataloader, args)


if __name__ == "__main__":
    main()