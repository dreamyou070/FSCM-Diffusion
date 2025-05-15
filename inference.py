import argparse
from torchvision import transforms
import torch
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import os
import cv2
import torch
from model.structuredlatent import CustomStructuredConv
import time
from utils.base import convert_to_np, numpy2torch
from model.unet_model import UNet2DConditionModel
from model.climatecontrol_pipeline import ClimateControlPipeline

def main(args) :

    print(f' step 1. load model')
    device = args.device

    model_id = args.base_model_id
    weight_dtype = torch.float16
    base = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = ClimateControlPipeline.from_pretrained(model_id, torch_dtype=weight_dtype).to(device)
    resume_from_checkpoint = 'checkpoint-' + str(id)
    start = time.time()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(base, resume_from_checkpoint), subfolder="unet",
                                                revision=args.revision, torch_dtype=weight_dtype).to(device)
    end = time.time()
    print(f"UNet loaded in {end - start:.2f} seconds")
    unet.eval()
    pipe.unet = unet


    if args.use_fused_conditionmap :
        if args.test_image_condition :
            print(f' making structure_obj!')
            structure_obj = CustomStructuredConv(in_channels=5, out_channels=1)
        else :
            structure_obj = CustomStructuredConv(in_channels=5, )
        state_dict = torch.load(os.path.join(base, resume_from_checkpoint,"structured_obj.bin"), map_location='cpu')
        structure_obj.load_state_dict(state_dict)
        pipe.set_condition_model(structure_obj.to(device))

    print(f' (1.2) save_folder')
    save_folder = os.path.join(args.output_dir, f"validation_{resume_from_checkpoint}_more")

    print(f' (1.3) inference')
    generator = torch.Generator("cuda").manual_seed(args.seed)
    style_prompt = args.style_prompt
    img_dir = args.input_image
    original_image = Image.open(img_dir).convert("RGB").resize((args.resolution, args.resolution),)
    # 2)
    depth_dir = args.depth_map
    depthmap_image = Image.open(depth_dir).convert("L")
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=False,
                                       do_resize=True, do_convert_grayscale=True)
    # 3)
    normal_transform = transforms.RandomCrop(args.resolution)
    def load_images(image_list, channel=3):
        arrs = [convert_to_np(Image.open(img_path), args.resolution) for img_path in image_list]
        arrs = np.concatenate(arrs)
        tensor = torch.tensor(arrs)
        tensor = 2 * (tensor / 255) - 1
        return normal_transform(tensor).reshape(-1, channel, args.resolution, args.resolution)
    normalmap = load_images([args.normal_map]).to(device = device, dtype = weight_dtype)
    normal_map_latent = pipe.vae.encode(normalmap).latent_dist.mode()

    # 4)
    level = args.level
    class_label = int(lavel) + args.base_label_number #


    class_labels = torch.LongTensor([int(class_label)]).to(device=device)
    # 1)
    img_base_name = os.path.basename(img_dir)
    original_image.save(os.path.join(save_folder, img_base_name))
    img_base_name, ext = os.path.splitext(img_base_name)
    num_inference_steps = args.num_inference_steps

    image_guidance_scales = [1.5]
    prompt = style_prompt + prompt
    if args.use_depthmap :
        edited_image = pipe(prompt = prompt,                     # 1) concept key word
                            image=original_image,                # 2) start image
                            class_labels=class_labels,           # 3) label (long tensor)
                            depth_image=depthmap_image,          # 4.1) depth_image
                            mask_processor=mask_processor,
                            normal_map_latent=normal_map_latent, # 4.2) normal_map
                            structure_guidance_scale = args.structure_guidance_scale,
                            use_fused_condition=args.use_fused_conditionmap,
                            num_inference_steps=num_inference_steps,
                            image_guidance_scale=image_guidance_scale,
                            generator=generator,).images[0]
    else :
        edited_image = pipe(prompt=prompt,  # 1) concept key word
                            image=original_image,  # 2) start image
                            class_labels=class_labels,  # 3) label (long tensor)
                            structure_guidance_scale=args.structure_guidance_scale,
                            num_inference_steps=num_inference_steps,
                            image_guidance_scale=image_guidance_scale,
                            generator=generator, ).images[0]

    save_prompt = prompt.replace(' ', '_')
    #save_name = f"{img_base_name}_label_{str(label)}_p_{save_prompt}_s_{num_inference_steps}_g_{guidance_scale}_structure_{args.structure_guidance_scale}{ext}"
    save_name = f"{img_base_name}_label_{str(label)}_p_{save_prompt}_ig_{image_guidance_scale}{ext}"
    edited_image.save(os.path.join(save_folder,save_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_model_id", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--start_id", type=int)
    parser.add_argument("--end_id", type=int)
    parser.add_argument("--pyramid_noise", action="store_true")
    parser.add_argument("--test_image_condition", action="store_true")
    parser.add_argument("--use_fused_conditionmap", action='store_true')
    parser.add_argument("--use_depthmap", action='store_true')
    parser.add_argument("--perlin_noise", action='store_true')
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--concept_word", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution for input images.")
    parser.add_argument("--output_dir", type=str, default="instruct-pix2pix-model",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--base_label_number", type=int)
    parser.add_argument("--prompt_folder", type=str)
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--use_noisy_structurelatent", action='store_true')
    parser.add_argument("--timewise_condition", action='store_true')
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of denoising steps during inference.")
    parser.add_argument("--image_guidance_scale", type=float, default=1.0,
                        help="Scale for image guidance.")
    parser.add_argument("--guidance_scales", type=int, nargs="+", default=[7],
                        help="List of guidance scales to test.")
    parser.add_argument("--structure_guidance_scale", type=float, default=1.5,) #
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--style_prompt", type=str, default="")
    parser.add_argument("--class_label_list", type=int, nargs="+", default=[0,1,2])
    args = parser.parse_args()
    main(args)