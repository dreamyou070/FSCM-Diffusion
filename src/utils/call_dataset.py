import datasets
from datasets import load_dataset
import os
from transformers import CLIPTextModel, CLIPTokenizer
from .base import convert_to_np
from PIL import Image
from torchvision import transforms
import torch
import numpy as np


DATASET_NAME_MAPPING = {"fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),}

def load_dataloader(args, tokenizer, accelerator):
    # --------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        data_files = {"train": os.path.join(args.train_data_dir, "**")}
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

    # --------------------------------------------
    # 2. Column 설정 (image, caption, label 등)
    # --------------------------------------------
    def get_column(name_arg, idx_fallback):
        if name_arg:
            if name_arg not in column_names:
                raise ValueError(f"{name_arg} must be one of: {', '.join(column_names)}")
            return name_arg
        return dataset_columns[idx_fallback] if dataset_columns else column_names[idx_fallback]

    original_image_column = get_column(args.original_image_column, 0)
    edit_prompt_column = get_column(args.edit_prompt_column, 1)
    edited_image_column = get_column(args.edited_image_column, 2)
    label_column = column_names[3]
    depthmap_column = column_names[4]
    normal_map_column = column_names[5]

    # --------------------------------------------
    # 3. Text Tokenizer
    # --------------------------------------------
    def tokenize_captions(captions):
        return tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

    # --------------------------------------------
    # 4. Image Transform
    # --------------------------------------------
    transform_list = []
    if args.center_crop:
        transform_list.append(transforms.CenterCrop(args.resolution))
    else:
        transform_list.append(transforms.RandomCrop(args.resolution))
    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    train_transforms = transforms.Compose(transform_list)

    # --------------------------------------------
    # 5. Preprocessing Function
    # --------------------------------------------
    def preprocess_train(examples):
        dataset_folder_name = str(args.dataset_name.split('/')[-1])
        base_dir = f'../../../data/diffusion/ClimateDiffusion/TrainData/dataconstruction/folder_train/{dataset_folder_name}'

        def load_images(image_list, channel=3):
            arrs = [convert_to_np(Image.open(os.path.join(base_dir, img_path)), args.resolution) for img_path in image_list]
            arrs = np.concatenate(arrs)
            tensor = torch.tensor(arrs)
            tensor = 2 * (tensor / 255) - 1
            return train_transforms(tensor).reshape(-1, channel, args.resolution, args.resolution)

        examples["original_pixel_values"] = load_images(examples[original_image_column])
        examples["edited_pixel_values"] = load_images(examples[edited_image_column])
        if args.use_normalmap:
            examples["normal_map_pixel_values"] = load_images(examples[normal_map_column])

        # Tokenize prompts
        examples["input_ids"] = tokenize_captions(list(examples[edit_prompt_column]))

        # Load depth images (as PIL Images)
        examples["depth_maps"] = [
            Image.open(os.path.join(base_dir, d)).convert('L') for d in examples[depthmap_column]
        ]
        examples["labels"] = list(examples[label_column])
        return examples

    # --------------------------------------------
    # 6. Apply Preprocessing
    # --------------------------------------------
    with accelerator.main_process_first():
        if args.max_train_samples:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # --------------------------------------------
    # 7. Collate Function
    # --------------------------------------------
    def collate_fn(batch):
        def stack(key):
            return torch.stack([ex[key] for ex in batch]).to(memory_format=torch.contiguous_format).float()

        result = {
            "original_pixel_values": stack("original_pixel_values"),
            "edited_pixel_values": stack("edited_pixel_values"),
            "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
            "labels": torch.tensor([int(ex["labels"]) + args.base_label_number for ex in batch])}

        if args.use_normalmap:
            result["normalmap_images"] = stack("normal_map_pixel_values")

        result["depthmap_values"] = [ex["depth_maps"] for ex in batch]  # list of PIL Images
        return result

    # --------------------------------------------
    # 8. DataLoader 생성
    # --------------------------------------------
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataset, train_dataloader

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids