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

def ssim(img1, img2, window_size=11, val_range=255, window=None, size_average=True, full=False):
    # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    L = val_range

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        # window should be at least 11x11
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret

def main():

    record_file = 'ssim.txt'

    print("🔧 step 1. SSIM...")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    load_images = lambda path, h, w: cv2.resize(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                                                ((w, h)))
    tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)


    print("🔧 step 2. directory")
    base_folder = 'folders'
    modes = ['ip2p','flux','ours','ours_lora']  # flux, ip2p, ours
    disasters = ['flood','snow']

    #modes = ['ours']  # flux, ip2p, ours
    #disasters = ['snow']

    for mode in modes :

        for disaster in disasters :
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
                total_count = 0

                files = sorted(os.listdir(origin_folder))
                # print("📁 Comparing images...")

                for file in tqdm(files, desc="Processing"):
                    name, ext = os.path.splitext(file)
                    origin_path = os.path.join(origin_folder, file)

                    for level in range(1, 5):
                        level_path_pattern = os.path.join(level_folders[level], f"{name}*")
                        print(f'level_path_pattern {level_path_pattern}')
                        matched_files = glob.glob(level_path_pattern)


                        if matched_files:
                            #print(f"⚠️ Missing level {level} image for {name}")
                            #continue
                        #else :
                            org_img = tensorify(load_images(origin_path, 512, 512))
                            lev_img = tensorify(load_images(matched_files[0], 512, 512))
                            score = ssim(org_img, lev_img)
                            print(f'{name} score = {score}')
                            total_ssim += score.item()
                            total_count += 1

                avg_ssim = total_ssim / total_count if total_count > 0 else 0.0
                print(f"\n✅ Average SSIM for {mode} ({disaster}): {avg_ssim:.4f}")
                with open(record_file, 'a') as f:
                    f.write(f" {disaster} | {mode} | {avg_ssim:.4f}\n")

if __name__ == '__main__':
    main()
