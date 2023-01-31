import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from data import CondLDM
from einops import rearrange
from omegaconf import OmegaConf
from taming.models import vqgan
from accelerate import Accelerator
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

os.environ["NCCL_P2P_DISABLE"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type = int, default = 24)
parser.add_argument("--split", type = str, default = "test", help = "Path to eval test data")
parser.add_argument("--config", type = str, default = None, help = "Path to config file")
parser.add_argument("--checkpoint", type = str, default = None, help = "Path to checkpoint")
parser.add_argument("--save_image_size", type = int, default = 64)
parser.add_argument("--save_image_gen", type = str, default = None, help = "Path saved generated images")

'''
accelerate launch --num_cpu_threads_per_process 8 --main_process_port 9876 conditional_ldm.py --config cin256-v2.yaml --checkpoint /home/data/ckpts/hbansal/ldm/model.ckpt --save_image_gen /home/data/datasets/ImageNet100/CondGeneration/
'''
args = parser.parse_args()

accelerator = Accelerator()
os.makedirs(os.path.join(args.save_image_gen, args.split), exist_ok = True)

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    return model


def get_model():
    config = OmegaConf.load(args.config)  
    model = load_model_from_config(config, args.checkpoint)
    return model

def generate_images(model, sampler, dataloader, args):
    
    model, sampler, dataloader = accelerator.prepare(model, sampler, dataloader)
    model = model.to(accelerator.device)
    model.eval()

    ## Hyperparameters
    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   # for unconditional guidance

    model = model.module
    with torch.no_grad():
        with model.ema_scope():
            for class_indices, class_labels, folder_names in tqdm(dataloader):
                for folder_name in folder_names:
                    os.makedirs(os.path.join(args.save_image_gen, args.split, folder_name), exist_ok = True)
                indices = list(filter(lambda x: not os.path.exists(os.path.join(args.save_image_gen, args.split, folder_names[x], str(class_indices[x].item()) + ".png")), range(len(class_indices))))
                if len(indices) == 0: continue
                class_indices = [class_indices[i] for i in indices]
                class_labels  = [class_labels[i] for i in indices] 
                folder_names  = [folder_names[i] for i in indices]
                class_labels  = torch.tensor(class_labels)
                uc = model.get_learned_conditioning(
                                        {model.cond_stage_key: torch.tensor(len(indices)*[1000]).to(model.device)}
                                        )
                c = model.get_learned_conditioning({model.cond_stage_key: class_labels.to(model.device)})
                print(len(indices))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                    conditioning=c,
                                    batch_size=len(indices),
                                    shape=[3, args.save_image_size, args.save_image_size],
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc, 
                                    eta=ddim_eta)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                samples = 255. * rearrange(x_samples_ddim, 'b c h w -> b h w c').cpu().numpy()
                for index in range(len(samples)):
                    im = Image.fromarray(samples[index].astype(np.uint8))
                    im.save(os.path.join(args.save_image_gen, args.split, folder_names[index], str(class_indices[index].item()) + ".png"))
def main():
    model = get_model()
    sampler = DDIMSampler(model)

    dataset = CondLDM(num_images_per_class = 1300 if args.split == 'train' else 50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size)

    generate_images(model, sampler, dataloader, args)

if __name__ == "__main__":
    main()