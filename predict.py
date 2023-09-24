# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
import qrcode
import random
from PIL import Image
from typing import List
from PIL.Image import LANCZOS
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
BASE_CACHE = "model-cache"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
IMG_CACHE = "img-cache"

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def resize_for_condition_image(input_image, width, height):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(min(width, height)) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
            cache_dir=VAE_CACHE,
        )
        self.controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            torch_dtype=torch.float16,
            cache_dir=CONTROL_CACHE,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=self.controlnet,
            vae=self.vae,
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir=BASE_CACHE,
        ).to("cuda")

    def generate_qrcode(self, qr_code_content, background, border, width, height):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=border,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color=background)
        qrcode_image = resize_for_condition_image(qrcode_image, width, height)
        return qrcode_image

    def predict(
        self,
        prompt: str = Input(description="The prompt to guide QR Code generation."),
        qr_code_content: str = Input(
            description="The website/content your QR Code will point to.",
            default=None
        ),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=20, le=100, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=-1),
        width: int = Input(description="Width out the output image", default=768),
        height: int = Input(description="Height out the output image", default=768),
        num_outputs: int = Input(
            description="Number of outputs", ge=1, le=4, default=1
        ),
        image: Path = Input(
            description="Input image. If none is provided, a QR code will be generated",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original unet.",
            ge=0.0,
            le=4.0,
            default=1.0,
        ),
        border: int = Input(description="QR code border size", ge=0, le=4, default=1),
        qrcode_background: str = Input(
            description="Background color of raw QR code",
            choices=["gray", "white"],
            default="gray",
        ),
    ) -> List[Path]:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        print(f"Seed: {seed}")
        if image is None:
            if qrcode_background == "gray":
                qrcode_background = "#808080"
            image = self.generate_qrcode(
                qr_code_content, background=qrcode_background, border=border, width=width, height=height,
            )
        else:
            image = Image.open(str(image))
        out = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            image=[image] * num_outputs,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"/tmp/output-{i}.png"
            image.save(fname)
            outputs.append(Path(fname))

        return outputs

