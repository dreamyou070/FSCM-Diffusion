import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse
import os

def fuse_lora_weights_into_base(model, lora_scale=1.0):
    for name, module in model.named_modules():
        if hasattr(module, 'lora_up') and hasattr(module, 'lora_down'):
            if hasattr(module, 'weight'):
                print(f"[FUSE] {name} - merging LoRA into base weight")

                W = module.weight.data
                A = module.lora_up.weight.data
                B = module.lora_down.weight.data

                if hasattr(module, 'lora_alpha'):
                    scale = module.lora_alpha / A.shape[1]
                else:
                    scale = 1.0

                delta = (A @ B) * (lora_scale * scale)

                if W.dim() == 4 and delta.dim() == 2:
                    delta = delta.view_as(W)

                module.weight.data += delta

                # ✅ Remove LoRA components
                del module.lora_up
                del module.lora_down
                if hasattr(module, 'lora_alpha'):
                    del module.lora_alpha

    return model

def main(args):
    print("Step 1: Load base pipeline")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,             # avoid half precision errors during fuse
        low_cpu_mem_usage=False                # ensure weights are loaded, not meta
    )

    print("Step 2: Load LoRA weights")
    pipeline.load_lora_weights("./pretrained_model",
                               weight_name="Reflections.safetensors")

    print("Step 3: Move UNet to GPU and fuse")
    pipeline.unet.to("cuda")
    pipeline.unet = fuse_lora_weights_into_base(pipeline.unet)

    print("Step 4: Check if LoRA weights still remain (should not)")
    for k in pipeline.unet.state_dict().keys():
        if 'lora' in k or 'base_layer' in k:
            print(f"❌ Leftover LoRA key: {k}")
    print("✅ LoRA fuse complete.")

    print("Step 5: Save fused pipeline")
    pipeline.save_pretrained(args.save_dir, safe_serialization=True)
    print(f"✅ Saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./pretrained_model/fused_instruct_pix2pix")
    args = parser.parse_args()
    main(args)
