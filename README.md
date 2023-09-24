# cog-illusion-diffusion-hq

[![Replicate](https://replicate.com/andreasjansson/qrcode/badge)](https://replicate.com/andreasjansson/qrcode)

This is an implementation of [Monster Labs' QR code control net](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="(masterpiece:1.4), (best quality), (detailed), Medieval village scene with busy streets and castle in the distance" -i image=@spiral.png

Output:
![Stable diffusion](output.png)