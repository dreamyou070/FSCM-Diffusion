import argparse
import json
from datasets import load_dataset
import logging
import math
import os
from torchvision import transforms
import shutil
from pathlib import Path
from packaging import version
# 외부 라이브러리
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from utils.call_dataset import tokenize_captions
# Torch 및 관련 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
# HuggingFace accelerate
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# HuggingFace transformers & datasets
from utils.call_dataset import load_dataloader
# HuggingFace hub
from huggingface_hub import create_repo, upload_folder
# Diffusers 관련
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (check_min_version,
    deprecate,
    is_wandb_available,
    is_xformers_available,)
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.torch_utils import is_compiled_module
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
# 사용자 정의 파이프라인 및 모델
from model.climatecontrol_pipeline import ClimateControlPipeline
from model.unet_model import UNet2DConditionModel
# from model.structuredlatent import CustomStructuredConv
import wandb
import datasets

import numpy as np
import torch

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def numpy2torch(numpy_obj):
    torch_obj = torch.tensor(numpy_obj)
    return 2 * (torch_obj / 255) - 1

logger = get_logger(__name__, log_level="INFO")
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

DATASET_NAME_MAPPING = {"fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),}

def main(args):

    print(f' step 1. make wandb project')
    save_args = vars(args)
    os.makedirs(args.output_dir, exist_ok=True)
    args_save_dir = os.path.join(args.output_dir, "args.json")
    with open(args_save_dir, "w") as f:
        f.write(json.dumps(save_args, indent=2))

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use huggingface-cli login to authenticate with the Hub.")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_config=accelerator_project_config, )
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project_name, name=args.wandb_name, )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False


    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True,
                                  token=args.hub_token).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                 subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                revision=args.non_ema_revision)

    if args.use_lora:
        fused_model_dir = './models/water_reflection_applied_pix2pix'
        unet = UNet2DConditionModel.from_pretrained(fused_model_dir,
                                                    subfolder='unet',
                                                    torch_dtype=torch.float16, )

    # ------------------------------------------------------------------------------------------------------------- #
    original_state_dict = unet.state_dict()
    config_dict = dict(unet.config)
    config_dict['class_embeddings_concat'] = True
    config_dict['num_class_embeds'] = args.num_levels
    config_dict['projection_class_embeddings_input_dim'] = 1
    unet = UNet2DConditionModel(**config_dict)
    empty_state_dict = unet.state_dict()
    for key in empty_state_dict:
        if key in original_state_dict:
            empty_value = empty_state_dict[key]
            origin_value = original_state_dict[key]
            if empty_value.shape != origin_value.shape:
                min_shape = tuple(min(e, o) for e, o in zip(empty_value.shape, origin_value.shape))
                new_tensor = empty_value.clone()
                slices = tuple(slice(0, s) for s in min_shape)
                new_tensor[slices] = origin_value[slices]
                empty_state_dict[key] = new_tensor
            else:
                empty_state_dict[key] = origin_value
    unet.load_state_dict(empty_state_dict)
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    structure_obj = None

    if args.use_depthmap:
        mask_processor = VaeImageProcessor(vae_scale_factor=8,
                                           do_normalize=False,
                                           do_binarize=False,
                                           do_resize=True,
                                           do_convert_grayscale=True)

        in_channels = 9
        if args.use_normalmap:
            in_channels = 13
            if args.use_fused_conditionmap:
                in_channels = 9
                class CustomStructuredConv(nn.Module):
                    def __init__(self, in_channels, out_channels=1):
                        super(CustomStructuredConv, self).__init__()  # ✅ 반드시 호출
                        conv_in_padding = (3 - 1) // 2
                        self.structured_conv_in = nn.Conv2d(in_channels,
                                                            out_channels,
                                                            kernel_size=3,
                                                            padding=conv_in_padding)

                    def forward(self, x):
                        return self.structured_conv_in(x)
                structure_obj = CustomStructuredConv(in_channels=5)
    else:
        if args.use_normalmap:
            in_channels = 12
    out_channels = unet.conv_in.out_channels
    print(f' final in channels = {in_channels}, out_channels = {out_channels}')
    unet.register_to_config(in_channels=in_channels)
    with torch.no_grad():
        new_conv_in = nn.Conv2d(in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride,
                                unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in
    unet.register_to_config(in_channels=in_channels)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):  # ✅ unet만 save_pretrained
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, CustomStructuredConv):  # ✅ 구조 layer는 torch 방식
                        torch.save(model.state_dict(), os.path.join(output_dir, "structured_obj.bin"))

                    # pop weight after save
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()

                # Load UNet model (Diffusers style)
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

                # Load CustomStructuredConv model (PyTorch style)
                elif isinstance(model, CustomStructuredConv):
                    model.load_state_dict(torch.load(os.path.join(input_dir, "structured_obj.bin")))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name,
                               args.dataset_config_name,
                               cache_dir=args.cache_dir, )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    label_column = column_names[3]
    depthmap_column = column_names[4]
    if args.use_normalmap:
        normal_map_column = column_names[5]

    def tokenize_captions(captions):
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt")
        return inputs.input_ids

    # Preprocessing the datasets.


    train_transforms = transforms.Compose(
        [transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
         ])

    def preprocess_train(examples):
        dataset_folder_name = str(args.dataset_name.split('/')[-1])
        base_dir = f'../../../data/diffusion/ClimateDiffusion/TrainData/dataconstruction/folder_train_snow/{dataset_folder_name}'
        #base_dir = f'../../../data/diffusion/ClimateDiffusion/TrainData/dataconstruction' #/folder_train/{dataset_folder_name}'

        def preprocess_images(examples):
            original_images = np.concatenate(
                [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
                 examples[original_image_column]])
            edited_images = np.concatenate(
                [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
                 examples[edited_image_column]])

            images = np.stack([original_images, edited_images])
            images = torch.tensor(images)
            images = 2 * (images / 255) - 1
            return train_transforms(images)

        preprocessed_images = preprocess_images(examples)
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)
        examples["original_pixel_values"] = original_images  # [batch, 3, 512,512]
        examples["edited_pixel_values"] = edited_images
        if args.use_normalmap:
            normalmap_images = np.concatenate(
                [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
                 examples[normal_map_column]])
            normalmap_images = torch.tensor(normalmap_images)
            normalmap_images = 2 * (normalmap_images / 255) - 1
            normalmap_images = train_transforms(normalmap_images)
            normalmap_images = normalmap_images.reshape(-1, 3, args.resolution, args.resolution)
            examples['normal_map_pixel_values'] = normalmap_images
        # [2]
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)

        # [3]
        depth_list = []
        for depthmap_dir in examples[depthmap_column]:
            depth = Image.open(os.path.join(base_dir, depthmap_dir)).convert('L')
            depth_list.append(depth)
        # depth_list = [Image.open(os.path.join(base_dir, depthmap_dir)).convert('L') for depthmap_dir in examples[depthmap_column]]
        examples["depth_maps"] = depth_list  #
        # [4]
        examples["labels"] = list(examples[label_column])
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()

        # water_color_pixel_values = water_color_pixel_values.to(memory_format=torch.contiguous_format).float()
        # depthmap_values = torch.stack([example["depth_maps"] for example in examples]).to(memory_format=torch.contiguous_format).float() # batch, 1, h, w
        depthmap_values = []
        for example in examples:
            depthmap = example["depth_maps"]
            depthmap_values.append(depthmap)
        # print(f' [collate_fn] depthmap_values = {depthmap_values}')
        input_ids = torch.stack([example["input_ids"] for example in examples])

        for example in examples:
            label = example["labels"]

        # labels = [19 if int(example["labels"]) > 19 else int(example["labels"]) + args.base_label_number for example in examples]
        labels = []
        for example in examples:
            # label = int( example["labels"] / 2) + args.base_label_number
            label = int(example["labels"]) + args.base_label_number
            labels.append(label)

        labels = torch.tensor(labels)

        if args.use_normalmap:

            normalmap_pixel_values = torch.stack([example['normal_map_pixel_values'] for example in examples])
            return {"original_pixel_values": original_pixel_values,
                    "edited_pixel_values": edited_pixel_values,
                    "depthmap_values": depthmap_values,
                    "input_ids": input_ids,
                    "labels": labels,
                    "normalmap_images": normalmap_pixel_values}
        else:
            return {"original_pixel_values": original_pixel_values,
                    "edited_pixel_values": edited_pixel_values,
                    "depthmap_values": depthmap_values,
                    "input_ids": input_ids,
                    "labels": labels}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    #
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # optimizer_cls(list(unet.parameters()) + list(unet.structured_conv_in.parameters()),
    optimizer = optimizer_cls(unet.parameters(),
                              lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )
    if structure_obj is not None:
        optimizer = optimizer_cls(list(unet.parameters()) + list(structure_obj.parameters()),
                                  lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
                args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our accelerator.
    if args.use_fused_conditionmap:
        unet, optimizer, train_dataloader, lr_scheduler, structure_obj = accelerator.prepare(unet, optimizer,
                                                                                             train_dataloader,
                                                                                             lr_scheduler,
                                                                                             structure_obj)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer,
                                                                              train_dataloader, lr_scheduler)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # here problem ...
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        print(f' Start Global Step : {global_step}')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):

        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with (accelerator.accumulate(unet)):
                class_labels = batch["labels"].to(device=accelerator.device)

                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # [2] src
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                noise = torch.randn_like(latents)

                # [1] target
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # only image

                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                                      * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds
                    # depth_latent

                structured_latents = None
                if args.use_depthmap:
                    def prepare_depth_latents(depth_condition, height, width, dtype, device,
                                              do_classifier_free_guidance):
                        vae_scale_factor = 8
                        depth_latent = torch.nn.functional.interpolate(depth_condition,
                                                                       size=(height // vae_scale_factor,
                                                                             width // vae_scale_factor))
                        depth_latent = depth_latent.to(device=device, dtype=dtype)
                        depth_latent = torch.cat(
                            [depth_latent] * 2) if do_classifier_free_guidance else depth_latent
                        return depth_latent

                    depthmap = batch["depthmap_values"]  # list of pillow
                    depth_condition = mask_processor.preprocess(depthmap,
                                                                height=args.resolution,
                                                                width=args.resolution,
                                                                resize_mode='default')
                    depth_latent = prepare_depth_latents(depth_condition, args.resolution, args.resolution,
                                                         weight_dtype, accelerator.device, False)  # Batch, 1, 64,64
                    structured_latents = depth_latent

                if args.use_normalmap:
                    normal_map = batch["normalmap_images"].to(weight_dtype)
                    normal_map_latent = vae.encode(normal_map).latent_dist.mode()
                    if structured_latents is not None:
                        structured_latents = torch.cat([structured_latents, normal_map_latent], dim=1)
                    else:
                        structured_latents = normal_map_latent

                if structured_latents is not None:
                    if args.use_fused_conditionmap:
                        structured_latents = structure_obj(structured_latents)
                    original_image_embeds = torch.cat([original_image_embeds, structured_latents], dim=1)

                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                image_embeds = None
                added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else None
                model_pred = unet(concatenated_noisy_latents,
                                  timesteps,
                                  encoder_hidden_states,
                                  added_cond_kwargs=added_cond_kwargs,
                                  class_labels=class_labels,  # class labels
                                  return_dict=False, )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # logging
                if accelerator.is_main_process:
                    wandb.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the checkpoints_total_limit
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ checkpoints_total_limit - 1 checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break

        # validation on every epoch
        if accelerator.is_main_process:

            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())

            torch.cuda.empty_cache()

        # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for InstructPix2Pix + LoRA + noise + condition maps")

    # === [Model / Checkpoint Loading] ===
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_fused_model_id", type=str, )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    # === [Dataset & Input] ===
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--original_image_column", type=str, default="input_image")
    parser.add_argument("--edited_image_column", type=str, default="edited_image")
    parser.add_argument("--edit_prompt_column", type=str, default="edit_prompt")
    parser.add_argument("--prompt_folder", type=str)
    parser.add_argument("--use_label", action="store_true")
    parser.add_argument("--base_label_number", type=int, default=2)

    # === [Conditioning / Noise Control] ===
    parser.add_argument("--perlin_noise", action="store_true")
    parser.add_argument("--pyramid_noise", action="store_true")
    parser.add_argument("--second_perlin_noise", action="store_true")
    parser.add_argument("--adaptive_noise", action="store_true")
    parser.add_argument("--simple_noise", action="store_true")
    parser.add_argument("--noisy_source", action="store_true")
    parser.add_argument("--use_depthmap", action="store_true")
    parser.add_argument("--use_normalmap", action="store_true")
    parser.add_argument("--use_fused_conditionmap", action="store_true")
    parser.add_argument("--use_lora", action="store_true")

    # === [Training] ===
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # === [Optimizer] ===
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # === [Validation & Output] ===
    parser.add_argument("--output_dir", type=str, default="instruct-pix2pix-model")
    parser.add_argument("--val_image_url", type=str, default=None)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int, default=50)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)

    # === [Augmentation] ===
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")

    # === [Logging / WandB / Mixed Precision] ===
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default=None)

    # === [Distributed / Xformers / TF32] ===
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    # === [Hub] ===
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)

    # === [Other] ===
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_levels", type=int)

    args = parser.parse_args()

    # Sync local rank with env if needed
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity check
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either --dataset_name or --train_data_dir")

    # Inherit revision if not explicitly set for non-EMA
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    main(args)