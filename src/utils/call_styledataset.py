from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms

class StyleFolderDataset(Dataset):
    def __init__(self, image_folder, resolution=512, center_crop=True, use_normalmap=True, use_depthmap=True):
        parent_path = os.path.dirname(image_folder)

        self.image_folder = os.path.join(parent_path, "rgb")
        self.depth_folder = os.path.join(parent_path, "depth") if use_depthmap else None
        self.normal_folder = os.path.join(parent_path, "normal") if use_normalmap else None

        self.use_depthmap = use_depthmap
        self.use_normalmap = use_normalmap

        self.image_paths = sorted([
            os.path.join(self.image_folder, fname)
            for fname in os.listdir(self.image_folder)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        base_tf = [crop, transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1)]

        self.image_transform = transforms.Compose(base_tf)
        self.aux_transform = transforms.Compose([crop, transforms.ToTensor()])  # [0, 1] 스케일 유지

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.image_transform(image)

        # depth & normal 로드
        sample = {
            "original_pixel_values": image_tensor
        }

        #if self.use_depthmap:
        depth_path = os.path.join(self.depth_folder, filename)
        depth_img = Image.open(depth_path).convert("L")  # grayscale
        sample["depthmap_values"] = self.aux_transform(depth_img)

        #if self.use_normalmap:
        normal_path = os.path.join(self.normal_folder, filename)
        normal_img = Image.open(normal_path).convert("RGB")
        sample["normalmap_images"] = self.aux_transform(normal_img)

        return sample # "original_pixel_values", depthmap_values, normalmap_images


def build_style_dataloader(style_image_folder, batch_size=4, resolution=512, center_crop=True, num_workers=4, use_normalmap=True, use_depthmap=True):
    dataset = StyleFolderDataset(
        image_folder=style_image_folder,
        resolution=resolution,
        center_crop=center_crop,
        use_normalmap=use_normalmap,
        use_depthmap=use_depthmap
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset, dataloader
