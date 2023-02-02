# Generative Robustness

This repo contains the code for the experiments in the 'Leaving Reality to Imagination: Robust Classification via Generated Datasets' paper.


## Link to ImageNet-1K-G-v1 dataset

You can download the `ImageNet-1K-G-v1` dataset from [here](https://drive.google.com/drive/folders/1-jLyiJ_S-VZMS5zQNR6e1xDOAkAVJxvs?usp=share_link). Even though we discuss three variants of ImageNet-1K-G-v1 in the paper, we make generations using captions of the class labels (*SD-Labels*) public for novel usecases by the community.

Structure of the dataset looks like:

```
* train_captions.csv
* train (1000 folders)
    * n01440764 (1300 images)
        * image1.jpeg
        *  .
        * imageN.jpeg
    * .
    * .
* val_captions.csv 
* val (1000 images)
    * n01440764 (50 images)
        * image1.jpeg
        *  .
        * imageN.jpeg
    * .
    * .
```

## Data Generation Using Stable Diffusion

[Stable Diffusion](https://github.com/CompVis/stable-diffusion) is a popular text-to-image generative model. Most of the code is adapted from the very popular [diffusers](https://github.com/huggingface/diffusers) library from HuggingFace.

However, it might not be straightforward to generate images from Stable Diffusion on multiple GPUs. To that end, we use the [accelerate](https://huggingface.co/docs/accelerate/index) package from Huggingface.

### Requirements
 
- Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
- 64-bit Python 3.7+ installation. 
- We used 5 A6000s 24GB DRAM GPUs for generation.

### Setup

```
1. git clone https://github.com/Hritikbansal/leaving_reality_to_imagination.git
2. cd leaving_reality_to_imagination
3. conda env create -f environment.yml
4. pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html (Replace based on your computer's hardware)
5. accelerate config
- This machine
- multi-GPU
- (How many machines?) 1
- (optimize with dynamo?) NO
- (Deepspeed?) NO
- (FullyShardedParallel?) NO
- (MegatronLM) NO
- (Num of GPUs) 5
- (device ids) 0,1,2,3,4
- (np/fp16/bp16) no
```

### Files

1. [generate_images_captions](generation/generate_images_captions.py) generates the images conditioned on the diverse text prompts (__SD Labels__).
2. [generate_images](generation/generate_images.py) generates the images conditioned on the images (__SD Images__).
3. [generate_images_i2i](generation.generate_images_i2i.py) generates the images conditioned on the encoded images and text (__SD Labels and Images__).
4. [conditional ldm](generation/conditional_ldm.py) generates images from the class-conditional latent diffusion model. You can download the model ckpt from [Stable Diffusion](https://github.com/CompVis/stable-diffusion) repo.

Move the [classes.py](generation/classes.py) and [folder_to_class.csv](generation/folder_to_class.csv) to the `imagenet_dir`.

### Commands

```
accelerate launch --num_cpu_threads_per_process 8 -m generation.generate_images_captions --batch_size 8 --data_dir <imagenet_dir> --save_image_gen <save dir> --diversity --split val
```

```
accelerate launch --num_cpu_threads_per_process 8 -m generation.generate_images --batch_size 2 --eval_test_data_dir <imagenet_dir> --save_image_gen <save dir> --split val
```

```
accelerate launch --num_cpu_threads_per_process 8 -m generation.generate_images_i2i --batch_size 12 --data_dir <imagenet_dir> --save_image_gen <save dir> --split val --diversity
```

```
accelerate launch --num_cpu_threads_per_process 8 conditional_ldm.py --config cin256-v2.yaml --checkpoint <model checkpoint> --save_image_gen <save dir>
```


## Training ImageNet Models Using FFCV

We suggest the users to create a separate FFCV conda environment for training ImageNet models.

### Preparing the Dataset
Following the ImageNet training pipeline of [FFCV](https://github.com/libffcv/ffcv-imagenet) for ResNet50, generate the dataset with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset)):

```bash
# Required environmental variables for the script:
cd train/
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```
Note that we prepare the dataset with the following FFCV configuration:
* ResNet-50 training: 50% JPEG 500px side length (*train_500_0.50_90.ffcv*)
* ResNet-50 evaluation: 0% JPEG 500px side length (*val_500_uncompressed.ffcv*)

- We have made some custom edits to [write_imagenet.py](train/write_imagenet.py) to generate augmented imagenet data

### Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_imagenet.py --config-file resnet_configs/resnext50.yaml --data.train_dataset=<path to train ffcv> --data.val_dataset=<path to validation ffcv> --data.num_workers=8 --logging.folder=<logging folder> --model.num_classes=100 (if imagenet 100) --training.distributed=1 --dist.world_size=5 
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_imagenet.py --config-file resnet_configs/rn18_88_epochs.yaml --data.train_dataset=<path to train ffcv> --data.val_dataset=<path to validation ffcv> --data.num_workers=8 --training.path=<path to final_weights.pt> --model.num_classes=1000 --training.distributed=1 --dist.world_size=5 --training.eval_only=1
```

### Note

1. Since ImageNet-R and ObjectNet do not share all the classes with ImageNet1K, we use an additional `validation.imr` or `validation.obj` flag while evaluating on these datasets.
2. [create_imagenet_subset](utils/create_imagenet_subset.py) is used to create the random subset containing 100 classes. [mappings](utils/mappings/) contains the relevant `imagenet100.txt` file.

## Natural Distribution Shift Datasets

We use (a) ImageNet-Sketch, (b) ImageNet-R, (c) ImageNet-V2, and (d) ObjectNet in our work. The users can navigate to their respective sources to download the data.

1. [rename_imagenetv2](utils/rename_imagenet_v2.py) renames the imagenetv2 folders that are original named based on indices 0-1000 to original imagenet folder names n0XXXXX.
2. [subset_objectnet_im](utils/subset_objectnet_im.py) is used to create a subset of ObjectNet classes that overlap with ImageNet-100/1000. 


