#!/bin/bash
# 백그라운드 실행 예시:
# nohup bash infer.sh > ./logs/infer_1305_batch5.log 2>&1 &
export MODEL_NAME="timbrooks/instruct-pix2pix"
# port
main_process_port=51074
# Accelerate 기반 InstructPix2Pix 추론 실행
accelerate launch \
  --config_file ../../accelerate_config/acc_0.yaml \
  --main_process_port $main_process_port \
  infer_style.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir '../result/4_1_flood/2_nerf_road_level0123_more_youtube_fscm_styleloss_only_teacher_distill' \
  --start_id 1800 --end_id 1851 \
  --prompt_folder '../../../data/diffusion/ClimateDiffusion/TestData/flood/test_flood.json' \
  --resolution 512 --seed 42 --base_label_number 0 --use_fused_conditionmap --style_prompt "reflection" \
  --class_label_list 0 1 2 3 4

# --concept_word "reflection"
#  --use_fused_conditionmap --dropout_prob 0.2 --test_image_condition
# nohup bash infer.sh
# if test_image_condition and dropout_prob is not None and  > 0 :