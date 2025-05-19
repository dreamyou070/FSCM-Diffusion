<<<<<<< HEAD
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import os
import lpips
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
import glob
import cv2
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms



def main():
    record_file = 'lpips.txt'

    # 이미지 로드 및 전처리 함수
    def load_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS는 이미지 크기가 같아야 함
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0)  # (1, 3, H, W)

    print("🔧 step 1. Load LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex')  # 'vgg' or 'squeeze'도 가능


    print("🔧 step 2. Set directory paths")
    base_folder = 'folders'
    modes = ['ip2p','flux','ours','ours_lora']
    disasters = ['flood','snow']

    for mode in modes:
        for disaster in disasters:
            if mode == 'ours_lora' and disaster == 'snow':
                continue

            print(f'▶️ Processing mode: {mode} | disaster: {disaster}')
            mode_folder = os.path.join(base_folder, mode)
            disaster_folder = os.path.join(mode_folder, disaster)
            origin_folder = os.path.join(base_folder, f'origin_{disaster}')
            level_folders = {
                1: os.path.join(disaster_folder, f'level1'),
                2: os.path.join(disaster_folder, f'level2'),
                3: os.path.join(disaster_folder, f'level3'),
                4: os.path.join(disaster_folder, f'level4'),
            }

            total_lpips = 0.0
            total_count = 0

            files = sorted(os.listdir(origin_folder))

            for file in tqdm(files, desc="Processing"):
                name, ext = os.path.splitext(file)
                origin_path = os.path.join(origin_folder, file)
                org_img = load_image(origin_path)

                for level in range(1, 5):
                    level_path_pattern = os.path.join(level_folders[level], f"{name}*")
                    matched_files = glob.glob(level_path_pattern)
                    if matched_files:
                        lev_img = load_image(matched_files[0])
                        # LPIPS 계산
                        lpips_distance = loss_fn(org_img, lev_img)
                        total_lpips += lpips_distance.item()
                        total_count += 1

            avg_lpips = total_lpips / total_count if total_count > 0 else 0.0

            print(f"✅ Average LPIPS for {mode} ({disaster}): {avg_lpips:.4f}")

            with open(record_file, 'a') as f:
                f.write(f" {disaster} | {mode} | LPIPS: {avg_lpips:.4f}\n")

if __name__ == '__main__':
    main()
=======
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import os
import lpips
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
import glob
import cv2
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms



def main():
    record_file = 'lpips.txt'

    # 이미지 로드 및 전처리 함수
    def load_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS는 이미지 크기가 같아야 함
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0)  # (1, 3, H, W)

    print("🔧 step 1. Load LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex')  # 'vgg' or 'squeeze'도 가능


    print("🔧 step 2. Set directory paths")
    base_folder = 'folders'
    modes = ['ip2p','flux','ours','ours_lora']
    disasters = ['flood','snow']

    for mode in modes:
        for disaster in disasters:
            if mode == 'ours_lora' and disaster == 'snow':
                continue

            print(f'▶️ Processing mode: {mode} | disaster: {disaster}')
            mode_folder = os.path.join(base_folder, mode)
            disaster_folder = os.path.join(mode_folder, disaster)
            origin_folder = os.path.join(base_folder, f'origin_{disaster}')
            level_folders = {
                1: os.path.join(disaster_folder, f'level1'),
                2: os.path.join(disaster_folder, f'level2'),
                3: os.path.join(disaster_folder, f'level3'),
                4: os.path.join(disaster_folder, f'level4'),
            }

            total_lpips = 0.0
            total_count = 0

            files = sorted(os.listdir(origin_folder))

            for file in tqdm(files, desc="Processing"):
                name, ext = os.path.splitext(file)
                origin_path = os.path.join(origin_folder, file)
                org_img = load_image(origin_path)

                for level in range(1, 5):
                    level_path_pattern = os.path.join(level_folders[level], f"{name}*")
                    matched_files = glob.glob(level_path_pattern)
                    if matched_files:
                        lev_img = load_image(matched_files[0])
                        # LPIPS 계산
                        lpips_distance = loss_fn(org_img, lev_img)
                        total_lpips += lpips_distance.item()
                        total_count += 1

            avg_lpips = total_lpips / total_count if total_count > 0 else 0.0

            print(f"✅ Average LPIPS for {mode} ({disaster}): {avg_lpips:.4f}")

            with open(record_file, 'a') as f:
                f.write(f" {disaster} | {mode} | LPIPS: {avg_lpips:.4f}\n")

if __name__ == '__main__':
    main()
>>>>>>> 0664c45 (Clean initial commit (no secrets))
