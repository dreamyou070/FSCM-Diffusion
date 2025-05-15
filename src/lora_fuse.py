import argparse
import json
from datasets import load_dataset
import os
from torchvision import transforms
from packaging import version
# 외부 라이브러리
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
# Torch 및 관련 모듈
import torch
import torch.utils.checkpoint
# HuggingFace accelerate
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
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
import wandb
import datasets
import numpy as np
import torch

def main(args):

    print(f' step 1. load base pipe')
    from diffusers import StableDiffusionInstructPix2PixPipeline
    MODEL_NAME = "timbrooks/instruct-pix2pix"
    weight_dtype = torch.float16
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_NAME,
                                                                      revision=args.revision,
                                                                      variant=args.variant,
                                                                      torch_dtype=weight_dtype,)
    lora_dir = './models/snow-style-richy-v1.safetensors'
    pipeline.load_lora_weights("./models",
                               weight_name="snow-style-richy-v1.safetensors")
    lora_loaded_unet = pipeline.unet

    save_dir = './models/snow_applied_pix2pix'

    def fuse_lora_weights_into_base(model, lora_scale=1.0):
        for name, module in model.named_modules():
            # LoRA 구조가 있는 모듈만 대상으로
            if hasattr(module, 'lora_up') and hasattr(module, 'lora_down'):
                if hasattr(module, 'weight'):
                    print(f"[FUSE] {name} - merging LoRA into base weight")

                    W = module.weight.data  # base weight
                    A = module.lora_up.weight.data
                    B = module.lora_down.weight.data

                    if hasattr(module, 'lora_alpha'):
                        scale = module.lora_alpha / A.shape[1]
                    else:
                        scale = 1.0

                    # LoRA delta = scale * (A @ B)
                    delta = (A @ B) * (lora_scale * scale)

                    # reshape if needed (e.g., conv2d)
                    if W.dim() == 4 and delta.dim() == 2:
                        delta = delta.view_as(W)

                    # Apply the fused delta to base weight
                    module.weight.data += delta
        return model

    org_unet = pipeline.unet
    new_unet = fuse_lora_weights_into_base(org_unet)
    pipeline.unet = new_unet

    # 4. Save the full pipeline (including config, unet, vae, text_encoder)
    print("Step 4: Save full standalone pipeline")
    pipeline.save_pretrained( save_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for InstructPix2Pix + LoRA + noise + condition maps")
    # === [Model / Checkpoint Loading] ===
    parser.add_argument("--lora_fused_model_id", type=str, )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    # === [Dataset & Input] ===
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


    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    main(args)