import torch
import cv2
import numpy as np
from pathlib import Path
import sys

class LoFTRMatcher:
    def __init__(self, weights_path: str = None):
        sys.path.append(str(Path(__file__).resolve().parent))
        from match_dir.src.loftr import LoFTR,default_cfg
        #from match_dir.src.config.default import get_cfg_defaults

        #self.cfg = get_cfg_defaults()
        #self.cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
        self.matcher = LoFTR(config=default_cfg)

        #if weights_path:
            
        self.matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
        
        # else:
        #     weights_path = torch.hub.download_url_to_file(
        #         'https://github.com/zju3dv/LoFTR/releases/download/weights/indoor_ds.ckpt',
        #         'loftr_indoor.ckpt')
        #     self.matcher.load_state_dict(torch.load(weights_path)['state_dict'])

        self.matcher = self.matcher.eval().cuda() if torch.cuda.is_available() else self.matcher.eval()

    def match(self, image_path1: str, image_path2: str, yolov5_centers: list = None, radius: int = 20):
        img0 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if img0 is None or img1 is None:
            raise ValueError("Could not load one or both images.")

        if img0.ndim == 3 and img0.shape[-1] == 1:
            img0 = np.squeeze(img0, axis=-1)
        if img1.ndim == 3 and img1.shape[-1] == 1:
            img1 = np.squeeze(img1, axis=-1)
        if img0.ndim == 3 and img0.shape[-1] in (3, 4):
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        if img1.ndim == 3 and img1.shape[-1] in (3, 4):
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img0 = torch.from_numpy(img0).unsqueeze(0).unsqueeze(0).float() / 255.
        img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float() / 255.

        if torch.cuda.is_available():
            img0, img1 = img0.cuda(), img1.cuda()

        batch = {'image0': img0, 'image1': img1}
        with torch.no_grad():
            self.matcher(batch)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        if yolov5_centers:
            yolov5_centers = np.array(yolov5_centers)
            filtered_mkpts0 = []
            filtered_mkpts1 = []
            filtered_conf = []

            for pt0, pt1, conf in zip(mkpts0, mkpts1, mconf):
                distances = np.linalg.norm(yolov5_centers - pt0, axis=1)
                if np.any(distances <= radius):
                    filtered_mkpts0.append(pt0.tolist())
                    filtered_mkpts1.append(pt1.tolist())
                    filtered_conf.append(float(conf))

            return {
                "filtered_matches_image0": filtered_mkpts0,
                "filtered_matches_image1": filtered_mkpts1,
                "match_confidence": filtered_conf
            }

        return {
            "matches_image0": mkpts0.tolist(),
            "matches_image1": mkpts1.tolist(),
            "match_confidence": mconf.tolist()
        }
