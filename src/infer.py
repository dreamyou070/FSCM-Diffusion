import argparse
from torchvision import transforms
from PIL import Image
import PIL, json
import requests
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

    for id in range(args.start_id, args.end_id,  50) :
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
        save_folder = os.path.join(args.output_dir, f"validation_{resume_from_checkpoint}_img_guidance_text_guidance")

        if args.test_second :
            save_folder = os.path.join(args.output_dir, f"validation_{resume_from_checkpoint}_img_guidance_text_guidance_add_leveling_before_T_all")

        if args.minus_class :
            save_folder = os.path.join(args.output_dir, f"validation_{resume_from_checkpoint}_img_guidance_text_guidance_add_leveling_before_T_all_minus_class")

        os.makedirs(save_folder, exist_ok=True)

        print(f' (1.3) inference')

        style_prompt = args.style_prompt
        prompt_folder = args.prompt_folder

        with open(prompt_folder, "r", encoding="utf-8") as f:
            datas = json.load(f)

        print(f' (1.4) prepare for infer')
        for i, item in enumerate(datas):
            data_base = f'../../data/diffusion/FSCM-Diffusion'
            img_dir = os.path.join(data_base, item['image'])
            original_image = Image.open(img_dir).convert("RGB").resize((args.resolution, args.resolution),)
            depth_dir = os.path.join(data_base, item['depthmap'])
            depthmap_image = Image.open(depth_dir).convert("L")
            mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=False,
                                               do_resize=True, do_convert_grayscale=True)

            normal_transform = transforms.RandomCrop(args.resolution)

            def load_images(image_list, channel=3):
                arrs = [convert_to_np(Image.open(img_path), args.resolution) for img_path in image_list]
                arrs = np.concatenate(arrs)
                tensor = torch.tensor(arrs)
                tensor = 2 * (tensor / 255) - 1
                return normal_transform(tensor).reshape(-1, channel, args.resolution, args.resolution)
            normalmap = load_images([os.path.join(data_base, item['normalmap'])]).to(device = device, dtype = weight_dtype)
            normal_map_latent = pipe.vae.encode(normalmap).latent_dist.mode()

            # 4)
            prompt_list = item['captions']
            for p_idx, prompt in enumerate(prompt_list) :
                label, prompt = prompt.split(',')
                class_label = int(label) + args.base_label_number #
                args.class_label_list = [int(e) for e in args.class_label_list]
                # 0,1,2,3,4
                # 4,5,6,7,8
                if class_label in args.class_label_list :
                    if args.test_12 :
                        class_label = class_label -1

                    neg_class_label = None
                    if args.minus_class :
                        neg_class_label = int(label) * -1 + args.base_label_number
                    # 0,-1,-2,-3,-4
                    # 4, 3, 2, 1, 0 -> optimizerd negative class
                    print(f'neg class label = {class_label}')
                    class_labels = torch.LongTensor([int(class_label)]).to(device=device)

                    img_base_name = os.path.basename(img_dir)

                    img_base_name, ext = os.path.splitext(img_base_name)
                    num_inference_steps = args.num_inference_steps

                    level_guidance_scales = [7.5]
                    image_guidance_scales = [1.5]

                    for image_guidance_scale in image_guidance_scales :

                       for level_guidance_scale in level_guidance_scales :

                            prompt = style_prompt + prompt

                            if not args.use_depthmap :
                                depthmap_image = None
                                mask_processor = None

                            if not args.use_normalmap :
                                normal_map_latent = None

                            start_timesteps = [200,300,400,500,700,900]
                            for start_timestep in start_timesteps :
                                generator = torch.Generator("cuda").manual_seed(args.seed)
                                edited_image = pipe(prompt = prompt,                     # 1) concept key word
                                                    image = original_image,                # 2) start image
                                                    class_labels=class_labels,           # 3) label (long tensor)
                                                    neg_class_label = neg_class_label,
                                                    depth_image=depthmap_image,          # 4.1) depth_image
                                                    mask_processor=mask_processor,
                                                    normal_map_latent=normal_map_latent, # 4.2) normal_map
                                                    use_fused_condition=args.use_fused_conditionmap,
                                                    num_inference_steps=num_inference_steps,
                                                    guidance_scale = level_guidance_scale,
                                                    image_guidance_scale=image_guidance_scale,
                                                    generator=generator,
                                                    do_prompt_control = args.do_prompt_control,
                                                    start_timestep = start_timestep,
                                                    do_negative_level_guidance = args.do_negative_level_guidance,
                                                    test_second = args.test_second,
                                                    reverse=args.reverse).images[0]
                                save_prompt = prompt.replace(' ', '_')

                                guidance_folder = os.path.join(save_folder, f"negative_from_{start_timestep}")
                                os.makedirs(guidance_folder, exist_ok=True)
                                save_name = f"{img_base_name}_label_{str(label)}_{save_prompt}_is_{image_guidance_scale}_g_{level_guidance_scale}{ext}"
                                edited_image.save(os.path.join(guidance_folder ,save_name))
                                original_image.save(os.path.join(guidance_folder, f"{img_base_name}{ext}"))

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
    parser.add_argument("--class_label_list", type=int, nargs="+", default=[0,1])
    parser.add_argument("--test_folder", type=str, default="")
    parser.add_argument("--disaster", type=str, default="") # args.test_foldername
    parser.add_argument("--test_foldername", type=str, default="")  # args.
    parser.add_argument("--save_name", type=str, default="")  # args. save_name
    parser.add_argument("--use_normalmap", action='store_true') # do_negative_label_guidance
    parser.add_argument("--do_negative_level_guidance", action='store_true')  #
    parser.add_argument("--test_12", action='store_true')  #
    parser.add_argument("--do_prompt_control", action='store_true')  # do_double_negative_label_guidance
    parser.add_argument("--do_double_negative_label_guidance", action='store_true')  #
    parser.add_argument("--do_third_negative_label_guidance", action='store_true')  # test_second
    parser.add_argument("--test_second", action='store_true')  #
    parser.add_argument("--minus_class", action='store_true')  #
    parser.add_argument("--reverse", action='store_true')  #
    args = parser.parse_args()
    main(args)