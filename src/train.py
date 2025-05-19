# 표준 라이브러리
import argparse
import json
import logging
import math
import os
import shutil
import accelerate
# 외부 유틸리티 라이브러리
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from packaging import version
# PyTorch 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
# Hugging Face - Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# Hugging Face - Transformers & Datasets
import transformers
import datasets
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
# Hugging Face - Diffusers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,  # 커스텀이면 아래처럼 분리
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    is_xformers_available,
)
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.torch_utils import is_compiled_module
# Hugging Face - Hub
from huggingface_hub import create_repo, upload_folder
# 사용자 정의 모듈
import wandb
from utils.base import convert_to_np, numpy2torch
from utils.call_dataset import tokenize_captions, load_dataloader
from utils.call_styledataset import build_style_dataloader
from model.climatecontrol_pipeline import ClimateControlPipeline
from model.unet_model import UNet2DConditionModel  # 만약 커스텀이면 여기 유지
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor


logger = get_logger(__name__, log_level="INFO")
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

DATASET_NAME_MAPPING = {"fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"), }


def compute_text_embeddings(tokenizer, text_encoder, prompt):
    def encode_prompt(text_encoder,
                      input_ids,
                      attention_mask):
        text_input_ids = input_ids.to(text_encoder.device)
        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
        max_length = tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return text_inputs

    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt)
        prompt_embeds = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask, )
    return prompt_embeds


def call_dataset(args, tokenizer, accelerator):
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir, )
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
    train_transforms = transforms.Compose(
        [transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x), ])

    def preprocess_train(examples):
        dataset_folder_name = str(args.dataset_name.split('/')[-1])
        # base_dir = f'../../../data/diffusion/ClimateDiffusion/TrainData/dataconstruction/folder_train/{dataset_folder_name}'
        base_dir = f'../../../data/diffusion/ClimateDiffusion/TrainData/dataconstruction/folder_train/{dataset_folder_name}'

        def preprocess_images(examples):

            #original_images = np.concatenate(
            #    [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
            #     examples[original_image_column]])
            original_images = np.concatenate([convert_to_np(image, args.resolution) for image in examples[original_image_column]])
            #edited_images = np.concatenate(
            #    [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
            #     examples[edited_image_column]])

            edited_images = np.concatenate([convert_to_np(image, args.resolution) for image in  examples[edited_image_column]])

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
            #normalmap_images = np.concatenate(
            #    [convert_to_np(Image.open(os.path.join(base_dir, image)), args.resolution) for image in
            #     examples[normal_map_column]])
            normalmap_images = np.concatenate([convert_to_np(image, args.resolution) for image in examples[normal_map_column]])
            normalmap_images = torch.tensor(normalmap_images)
            normalmap_images = 2 * (normalmap_images / 255) - 1
            normalmap_images = train_transforms(normalmap_images)
            normalmap_images = normalmap_images.reshape(-1, 3, args.resolution, args.resolution)
            examples['normal_map_pixel_values'] = normalmap_images
        # [2]
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions, tokenizer)

        # [3]
        depth_list = []
        for depthmap in examples[depthmap_column]:
            #depth = Image.open(os.path.join(base_dir, depthmap_dir)).convert('L')
            depth = depthmap.convert('L')
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
        depthmap_values = []
        for example in examples:
            depthmap = example["depth_maps"]
            depthmap_values.append(depthmap)
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = []
        for example in examples:
            #label = int(example["labels"]) + args.base_label_number
            if args.test_02 :
                if int(example["labels"]) == 0: label = 0
                else : label = 1

            if args.test_03 :
                if int(example["labels"]) == 0: label = 0
                else : label = 1

            if args.test_12 :
                if int(example["labels"]) == 1: label = 0
                else : label = 1

            if args.test_13 :
                if int(example["labels"]) == 1: label = 0
                else : label = 1

            if args.test_23 :
                if int(example["labels"]) == 2: label = 0
                else : label = 1


            labels.append(label)
        labels = torch.tensor(labels)
        if args.use_normalmap:
            normalmap_pixel_values = torch.stack([example['normal_map_pixel_values'] for example in examples])
        else:
            normalmap_pixel_values = None
        return {"original_pixel_values": original_pixel_values,
                "edited_pixel_values": edited_pixel_values,
                "depthmap_values": depthmap_values,
                "input_ids": input_ids,
                "labels": labels,
                "normalmap_images": normalmap_pixel_values}

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn,
                                                   batch_size=args.train_batch_size,
                                                   num_workers=args.dataloader_num_workers, )
    return train_dataset, train_dataloader

class CustomStructuredConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.structured_conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.structured_conv_in(x)

def change_unet_structure(unet, args):

    mask_processor = None  # 기본값 설정

    # Rebuild UNet with modified config
    original_state_dict = unet.state_dict()
    config_dict = dict(unet.config)
    config_dict['class_embeddings_concat'] = True
    config_dict['num_class_embeds'] = args.num_levels
    config_dict['projection_class_embeddings_input_dim'] = 1

    unet = UNet2DConditionModel(**config_dict)
    empty_state_dict = unet.state_dict()

    for key in empty_state_dict:
        if key in original_state_dict:
            if empty_state_dict[key].shape == original_state_dict[key].shape:
                empty_state_dict[key] = original_state_dict[key]
            else:
                min_shape = tuple(min(e, o) for e, o in zip(empty_state_dict[key].shape, original_state_dict[key].shape))
                slices = tuple(slice(0, s) for s in min_shape)
                new_tensor = empty_state_dict[key].clone()
                new_tensor[slices] = original_state_dict[key][slices]
                empty_state_dict[key] = new_tensor

    unet.load_state_dict(empty_state_dict)

    # Determine in_channels
    in_channels = 8
    if args.use_depthmap:
        in_channels = 9
        mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                           do_binarize=False, do_resize=True,
                                           do_convert_grayscale=True)
        if args.use_normalmap:
            in_channels = 13
    elif args.use_normalmap:
        in_channels = 12

    scnet = None
    if args.use_fused_conditionmap:
        in_channels = 9
        scnet = CustomStructuredConv(in_channels=5)

    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(in_channels, out_channels,
                                kernel_size=unet.conv_in.kernel_size,
                                stride=unet.conv_in.stride,
                                padding=unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    print(f'in change_unet_structure → scnet: {scnet}')
    return unet, scnet, mask_processor



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

        wandb.init(project=args.wandb_project_name,
                   name=args.wandb_name, )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    args.seed = 42
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                              revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                 revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
                                        variant=args.variant)
    print(f' (1.1) teacher unet')
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                revision=args.non_ema_revision)
    print(f' (1.3) style teacher model')
    sd_pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5",
                                                      torch_dtype=torch.float16)
    sd_pipe.load_lora_weights('/workspace/model/diffusion', weight_name="Reflections.safetensors", adapter_name='water_reflection_lora')
    sd_pipe.set_adapters(["water_reflection_lora"], adapter_weights=[1.0])
    sd_pipe.fuse_lora(adapter_names=["water_reflection_lora"], lora_scale=1.0)
    style_teacher_unet = sd_pipe.unet
    del sd_pipe
    style_teacher_unet.eval()

    print(f' (1.3.2) key concept embedding')
    unet, scnet, mask_processor = change_unet_structure(unet, args)
    print(f' *** scnet = {scnet}')

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    print(f' [student] scnet = {scnet}')
    print(f' [teacher] unet = {unet}')
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # 저장 대상만 등록: teacher_unet은 제외
        model_name_dict = {id(accelerator.unwrap_model(unet)): "unet",  # student_unet만 저장
                           id(accelerator.unwrap_model(scnet)): "structured_obj"}

        def save_model_hook(models, weights, output_dir):

            if not accelerator.is_main_process:
                return

            for i, model in enumerate(models):

                unwrapped_model = accelerator.unwrap_model(model)
                model_id = id(unwrapped_model)

                model_name = model_name_dict.get(model_id, None)

                print(f'*** model_name = {model_name}')

                if model_name is None:
                    print(f"[Skip] Unknown or excluded model at index {i}")
                    continue
                if isinstance(unwrapped_model, UNet2DConditionModel) and model_name == "unet":
                    save_path = os.path.join(output_dir, "unet")
                    print(f"[Save Hook] Saving {model_name} to {save_path}")
                    unwrapped_model.save_pretrained(save_path)

                if model_name == "structured_obj" :
                    save_path = os.path.join(output_dir, "structured_obj.bin")
                    print(f"[Save Hook] Saving {model_name} to {save_path}")
                    torch.save(unwrapped_model.state_dict(), save_path)

                if weights:
                    weights.pop()


        def load_model_hook(models, input_dir):

            for i in range(len(models)):

                model = models.pop()
                #unwrapped_model = accelerator.unwrap_model(model)
                model_id = id(model)
                model_name = model_name_dict.get(model_id, None)
                print(f'loading,model_name = {model_name}')
                if model_name is None:
                    print(f"[Skip] Unknown or excluded model at index {i}")
                    continue
                # Load UNet from 'unet/' directory
                if isinstance(model, UNet2DConditionModel) and model_name == "unet":
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                # Load structured object (e.g., SCNet)
                elif model_name == "structured_obj":
                    load_path = os.path.join(input_dir, "structured_obj.bin")
                    print(f"[Load Hook] Loading {model_name} from {load_path}")
                    state_dict = torch.load(load_path, map_location="cpu")
                    model.load_state_dict(state_dict)
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    print(f' step 2. call dataset')
    train_dataset, train_dataloader = call_dataset(args, tokenizer, accelerator)
    print(f' step 7. optimizer')
    if args.use_8bit_adam:
        try :
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if scnet is not None :
        trainable_param = list(unet.parameters()) + list(scnet.parameters())
    else :
        trainable_param = list(unet.parameters())

    optimizer1 = optimizer_cls(trainable_param, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
                    args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes)
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    lr_scheduler1 = get_scheduler(args.lr_scheduler, optimizer=optimizer1,
                                  num_warmup_steps=num_warmup_steps_for_scheduler,
                                  num_training_steps=num_training_steps_for_scheduler, )

    print(f' step 8. prepare')
    (unet, scnet, optimizer1,train_dataloader,lr_scheduler1) = accelerator.prepare(unet, scnet, optimizer1, train_dataloader,lr_scheduler1,)
    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
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

                # [1] target
                latents = vae.encode(
                    batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                # style_latents = vae.encode(style_batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode() * vae.config.scaling_factor
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # --------------------------------------------------------------------------------------------------------
                # [2.1] image condition
                # --------------------------------------------------------------------------------------------------------
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                    null_conditioning = text_encoder(tokenize_captions([""], tokenizer).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask,
                                                        null_conditioning,
                                                        encoder_hidden_states) # level 하고 같이 가도 좋을듯
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                                      * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds

                structured_latents = None
                if args.use_depthmap:
                    def prepare_depth_latents(depth_condition, height, width, dtype, device,
                                              do_classifier_free_guidance):
                        vae_scale_factor = 8
                        depth_latent = torch.nn.functional.interpolate(depth_condition,
                                                                       size=(height // vae_scale_factor,
                                                                             width // vae_scale_factor))
                        depth_latent = depth_latent.to(device=device, dtype=dtype)
                        depth_latent = torch.cat([depth_latent] * 2) if do_classifier_free_guidance else depth_latent
                        return depth_latent

                    # [1]
                    depth_condition = mask_processor.preprocess(batch["depthmap_values"], height=args.resolution,
                                                                width=args.resolution, resize_mode='default')
                    depth_latent = prepare_depth_latents(depth_condition, args.resolution, args.resolution,
                                                         weight_dtype, accelerator.device, False)  # Batch, 1, 64,64
                    structured_latents = depth_latent

                if args.use_normalmap:
                    normal_map_latent = vae.encode(batch["normalmap_images"].to(weight_dtype)).latent_dist.mode()
                    if structured_latents is not None:
                        structured_latents = torch.cat([structured_latents, normal_map_latent], dim=1)
                    else:
                        structured_latents = normal_map_latent

                if structured_latents is not None:
                    if args.use_fused_conditionmap:
                        structured_latents = scnet(structured_latents)
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
                total_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(total_loss)
                # Student UNet
                optimizer1.step()
                lr_scheduler1.step()
                optimizer1.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # logging
                if accelerator.is_main_process:
                    wandb.log({"train_loss": total_loss,},
                              step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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
            logs = {"train_loss": total_loss.detach().item(), "lr": lr_scheduler1.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break

        # validation on every epoch
        if accelerator.is_main_process:
            if ((args.val_image_url is not None)
                    and (args.validation_prompt is not None)
                    and (epoch % args.validation_epochs == 0)
            ):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                import copy
                unwrapped_unet = unwrap_model(unet)
                unwrapped_unet_copy = copy.deepcopy(unwrapped_unet)
                if args.use_ema:
                    ema_unet.restore(unet.parameters())
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
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
    parser.add_argument("--style_loss_lambda", type=float, default=0.1)

    parser.add_argument("--test_02", action="store_true", help="Run test case 02 (label: 0 if label==0 else 1)")
    parser.add_argument("--test_03", action="store_true", help="Run test case 03 (label: 0 if label==0 else 1)")
    parser.add_argument("--test_12", action="store_true", help="Run test case 12 (label: 0 if label==1 else 1)")
    parser.add_argument("--test_13", action="store_true", help="Run test case 13 (label: 0 if label==1 else 1)")
    parser.add_argument("--test_23", action="store_true", help="Run test case 13 (label: 0 if label==1 else 1)")
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