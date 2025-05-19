#!/bin/bash
# nohup bash infer.sh > ./logs/infer_flood_2_1000checkpoint.log 2>&1 &
export MODEL_NAME="timbrooks/instruct-pix2pix"
main_process_port=52193
disaster="flood"
config_file="acc_0.yaml"
filename="1_only_labeling_with_direction"

accelerate launch \
  --config_file ../accelerate_config/$config_file \
  --main_process_port $main_process_port \
  infer.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir "../result/$disaster/$filename" \
  --start_id 250 --end_id 251 \
  --prompt_folder "../../data/diffusion/FSCM-Diffusion/TestData/$disaster/test_$disaster.json" \
  --resolution 512 --seed 42 --base_label_number 0 --class_label_list 0 1 2 3 4 --disaster $disaster \
  --train_key_concept
#  5 6 7 8 --use_depthmap --do_negative_level_guidance --minus_class
#\  --minus_class   --use_depthmap --use_depthmap