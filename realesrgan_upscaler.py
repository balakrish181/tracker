import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# cuDNN safe settings to avoid conv plan errors
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class DermaRealESRGANx2:
    def __init__(self, model_path, fp32=True, gpu_id=None):
        """
        Fixed RealESRGAN x2 upscaler for dermaRealESRGAN_x2plus_v1.pth

        Args:
            model_path (str): Path to dermaRealESRGAN_x2plus_v1.pth
            fp32 (bool): Use full precision (safer), False = half precision
            gpu_id (int): GPU device ID or None for auto
        """
        self.netscale = 2
        self.outscale = 2
        self.fp32 = fp32
        self.gpu_id = gpu_id

        # Model architecture fixed for x2 RealESRGAN
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2
        )

        # Create upsampler
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            model=model,
            half=not fp32,
            gpu_id=gpu_id
        )

    def upscale(self, input_path, output_path=None):
        """Upscale an image from file path."""
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        output, _ = self.upsampler.enhance(img, outscale=self.outscale)

        if output_path:
            cv2.imwrite(output_path, output)
            return output_path
        return output


if __name__ == '__main__':
    upscaler = DermaRealESRGANx2(
        model_path='weights/dermaRealESRGAN_x2plus_v1.pth',
        fp32=True
    )
    upscaler.upscale('alix_mole.jpg', 'alix_mole_upscaled.jpg')
