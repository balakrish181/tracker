

import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN


class upscaler_esrgan:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(self.device, scale=4)
        self.model.load_weights('weights/4x-ESRGAN.pth', download=True)

    def upscale(self, image_path):
        image = Image.open(image_path).convert('RGB')
        sr_image = self.model.predict(image)
        return sr_image

    def upscale_image(self, image_path, save_path):
        sr_image = self.upscale(image_path)
        sr_image.save(save_path)


    
if __name__ == '__main__':
    upscaler = upscaler_esrgan()
    upscaler.upscale_image('alix.jpg', 'alix_upscaled.jpg')