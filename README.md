# leaving_reality_to_imagination
Github Repo to generate images from Stable Diffusion


This repo contains the code for the experiments in the 'Leaving Reality to Imagination: Robust Classification via Generated Datasets' paper.


## Data Generation Using Stable Diffusion

(Stable Diffusion)[https://github.com/CompVis/stable-diffusion] is a popular text-to-image generative model. Most of the code is adapted from the very popular (diffusers)[https://github.com/huggingface/diffusers] library from HuggingFace.

However, it might not be straightforward to generate images from Stable Diffusion on multiple GPUs. To that end, we use the (accelerate)[https://huggingface.co/docs/accelerate/index] package from Huggingface.

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

1. (generate_images_captions)[generation/generate_images_captions.py] generates the images conditioned on the diverse text prompts (__SD Labels__).
2. (generate_images)[generation/generate_images.py] generates the images conditioned on the images (__SD Images__).
3. (generate_images_i2i)[generation.generate_images_i2i.py] generates the images conditioned on the encoded images and text (__SD Labels and Images__).

### Commands