import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import os
import lpips_cal
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2


def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

def main():
    load_images = lambda path, h, w: cv2.resize(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                                                ((w, h)))
    tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)

    record_file = 'psnr.txt'
    print("🔧 step 2. directory")
    base_folder = 'folders'
    modes = ['ip2p', 'flux', 'ours', 'ours_lora']
    disasters = ['flood', 'snow']

    for mode in modes:
        for disaster in disasters:
            do = True
            if mode == 'ours_lora' and disaster == 'snow':
                do = False

            if do :
                print(f' mode {mode} disaster {disaster} processing ...')
                mode_folder = os.path.join(base_folder, mode)
                disaster_folder = os.path.join(mode_folder, disaster)
                origin_folder = os.path.join(base_folder, f'origin_{disaster}')
                level_folders = {
                    1: os.path.join(disaster_folder, f'level1'),
                    2: os.path.join(disaster_folder, f'level2'),
                    3: os.path.join(disaster_folder, f'level3'),
                    4: os.path.join(disaster_folder, f'level4'),
                }

                total_ssim = 0.0
                total_psnr = 0.0
                total_count = 0

                files = sorted(os.listdir(origin_folder))

                for file in tqdm(files, desc="Processing"):
                    name, ext = os.path.splitext(file)
                    origin_path = os.path.join(origin_folder, file)

                    for level in range(1, 5):
                        level_path_pattern = os.path.join(level_folders[level], f"{name}*")
                        matched_files = glob.glob(level_path_pattern)

                        if matched_files:
                            org_img = tensorify(load_images(origin_path, 512, 512))
                            lev_img = tensorify(load_images(matched_files[0], 512, 512))
                            psnr_score = calculate_psnr(org_img, lev_img)
                            total_psnr += psnr_score.item()
                            total_count += 1

                avg_psnr = total_psnr / total_count if total_count > 0 else 0.0
                print(f"\n✅ {mode} ({disaster}), PSNR: {avg_psnr:.4f}")
                with open(record_file, 'a') as f:
                    f.write(f"{disaster} | {mode} | PSNR: {avg_psnr:.4f}\n")
if __name__ == '__main__':
    main()