#!/usr/bin/env python3
#coding:utf-8

from torch import autocast

from tqdm import tqdm, trange
import numpy as np
from PIL import Image

import torch
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from torch import autocast
from einops import rearrange
from contextlib import contextmanager, nullcontext

def txt2img(
            model=None,
            prompt=None,
            start_code=None,
            scale=7,
            ddim_steps=50,
            H=512,
            W=512,
            batch_size=1,
            plms=True,
            tqdm_progress_disabled=True,
            ):
    
    ddim_eta = 0.0 #ddim eta (eta=0.0 corresponds to deterministic sampling
    C = 4 #latent channels
    f = 8 #downsampling factor


    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)


    assert prompt is not None
    data = batch_size * prompt

    images = []
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data", disable=tqdm_progress_disabled):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)


                    for x_sample in x_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        images.append(img)
    return images
