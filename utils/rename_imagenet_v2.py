import os
import argparse
import pandas as pd 
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNetV2 Rename')
    parser.add_argument('--imagenetv2_path', metavar='IMAGENET_DIR',
                        help='path to the existing full ImageNetV2 dataset')
    parser.add_argument('--datafile', help='path to labels file containing class name and folder name mapping')
    args = parser.parse_args()

    folder_names = os.listdir(args.imagenetv2_path)

    df = pd.read_csv(args.datafile)

    classnames = [classname.split('/')[1] for classname in list(df['image'])]
    classnames = [i for n, i in enumerate(classnames) if i not in classnames[:n]]

    for i in tqdm(range(1000)):
        os.rename(os.path.join(args.imagenetv2_path, f'{i}'), os.path.join(args.imagenetv2_path, f'{classnames[i]}'))