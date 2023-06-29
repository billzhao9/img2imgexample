import torch
from torch import autocast
from diffusers import StableDiffusionInstructPix2PixPipeline,UniPCMultistepScheduler
import base64
import io
import json
import os
import time
import urllib.request
import requests
import sys
from pathlib import Path
from typing import Optional, Union,List
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import re
import logging
from pydantic import BaseModel

device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "/persistent-storage/"
init_model = "timbrooks/instruct-pix2pix"


def download_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image



class Item(BaseModel):
    command: Optional[str]
    model_id: Optional[str]
    images_data: Optional[Union[str, List[str]]] = []
    # prompt params
    prompt: Optional[Union[str, List[str]]]
    negative_prompt:Optional[Union[str, List[str]]]
    #img2img no width and height params!
    num_inference_steps: Optional[int] = 25
    num_images_per_prompt: Optional[int] = 1
    seed: Optional[int]
    guidance_scale: Optional[int] = 9
    image_guidance_scale:Optional[int] = 1.5
    
# init model weight
def init_model():
    print("init model")
    global model
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(init_model, torch_dtype=torch.float16, cache_dir=cache_dir).to(device)
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

    model.enable_xformers_memory_efficient_attention()
    model.enable_attention_slicing()

init_model()

def predict(item, run_id, logger, binaries=None):
    logger.info("remote call recieved")
    global model
    item = Item(**item)

    model_id = item.model_id
    images_data = item.images_data

    #prompt related 
    prompt = item.prompt
    negative_prompt = item.negative_prompt
    guidance_scale = item.guidance_scale
    seed = item.seed
    num_inference_steps = item.num_inference_steps
    num_images_per_prompt = item.num_images_per_prompt
    image_guidance_scale = item.image_guidance_scale

    final_images = []

    # pre-deal imgs
    if images_data:
        if isinstance(images_data, str):
            logger.info("received image_data")
            if images_data.startswith("http://") or images_data.startswith("https://"):
                image = download_image(images_data)
            else:
                image = Image.open(BytesIO(base64.b64decode(images_data))).convert("RGB")
            
            logger.info(image)

            final_images.append(image)
            logger.info("final_mask_images")

        elif isinstance(images_data, list) and all(isinstance(item, str) for item in images_data):
            logger.info("received images_data array")
            temp_images = []
            for temp_image_data in images_data:
                if temp_image_data.startswith("http://") or temp_image_data.startswith("https://"):
                    temp_image = download_image(temp_image_data)
                else:
                    temp_image = Image.open(BytesIO(base64.b64decode(temp_image_data))).convert("RGB")
        
                temp_images.append(temp_image)

            final_images = temp_images

    #custom model weight if need
    if model_id:
        logger.info("reload new model")
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir).to(device)
        model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

        model.enable_xformers_memory_efficient_attention()
        model.enable_attention_slicing()

    generator = None
    if seed != None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(int(seed))
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    if images_data == None:
        return {'message': "No image data provided"}
    
    #INFERENCE
    with autocast("cuda"):
        images = model(prompt, negative_prompt=negative_prompt, image=final_images, num_inference_steps=num_inference_steps,                image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, generator=generator).images

        if images is not None:
            logger.info("finalizing return images")       
            finished_images = []
            for image in images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            return finished_images
        else:
            return {"result": False}