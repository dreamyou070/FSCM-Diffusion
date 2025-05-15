#!/bin/bash
# 백그라운드 실행 예시:
# nohup bash infer.sh > ./logs/1_only_labeling.log 2>&1 &

export MODEL_NAME="timbrooks/instruct-pix2pix"
main_process_port=55025

accelerate launch \
  --config_file ../../accelerate_config/acc_2.yaml \
  --main_process_port $main_process_port \
  infer.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir '../result/4_1_flood/5_fscm' \
  --start_id 900 --end_id 901 \
  --prompt_folder '../../../data/diffusion/ClimateDiffusion/TestData/flood/test_flood2.json' \
  --resolution 512 --seed 42 --base_label_number 0 --use_depthmap --use_fused_conditionmap \
  --class_label_list 0 1 2 3 4

# --use_depthmap
# --use_depthmap --style_prompt "reflection"
# --prompt_folder '../../../data/diffusion/ClimateDiffusion/TestData/flood/test_flood.json' \