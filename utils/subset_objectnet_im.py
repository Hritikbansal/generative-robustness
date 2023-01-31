import os
import json
import shutil
import argparse
from tqdm import tqdm

def create_subset(args):

    dirname = 'Objectnet/val'
    os.makedirs(os.path.join(args.im100_root, dirname), exist_ok = True)

    with open(args.folder_to_label, 'r') as f:
        folder_to_label = json.load(f)
        label_to_folder = {v:k for k,v in folder_to_label.items()}
    
    with open(args.objectnet_to_im100_folder, 'r') as f:
        objectnet_to_im100_folder = json.load(f)

    for label in tqdm(objectnet_to_im100_folder):
        folder_name = label_to_folder[label]
        images = os.listdir(os.path.join(args.objectnet_root, folder_name))
        im100_folder_name = objectnet_to_im100_folder[label]
        os.makedirs(os.path.join(args.im100_root, dirname, im100_folder_name), exist_ok = True)
        print(im100_folder_name)
        for image in images:
            shutil.copy(os.path.join(args.objectnet_root, folder_name, image), os.path.join(args.im100_root, dirname, im100_folder_name))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNetV2 Rename')
    parser.add_argument('--objectnet_root', type=str, default=None)
    parser.add_argument('--im100_root', type=str, default=None)
    parser.add_argument('--folder_to_label', type=str, default=None)
    parser.add_argument('--objectnet_to_im100_folder', type=str, default=None)
    args = parser.parse_args()

    create_subset(args)