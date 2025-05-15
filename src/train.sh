# 백그라운드 실행 예시:

# nohup bash train.sh > ./logs/train_fused_no_distill.log 2>&1 &

export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_ID="ParkSY/snow_dataset_nerf_youtube"
export main_process_port=55007
export ACCELERATE_FIND_UNUSED_PARAMETERS=true

accelerate launch \
  --config_file ../../accelerate_config/acc_1.yaml \
  --main_process_port $main_process_port \
  --mixed_precision="fp16" \
  train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_ID \
  --base_label_number 0 \
  --num_levels 5 \
  --resolution=512 \
  --random_flip \
  --train_batch_size=10 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=150000 \
  --learning_rate=5e-05 \
  --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --wandb_project_name="cctv_project" \
  --wandb_name="7_fscm_lora_fused_no_distill" \
  --output_dir="../result/4_1_flood/7_fscm_lora_fused_no_distill" \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=50 \
  --seed 42 \
  --conditioning_dropout_prob=0.05 \
  --use_depthmap --use_normalmap --use_fused_conditionmap --use_lora


# --conditioning_dropout_prob=0.05
# --use_depthmap \
# --use_normalmap \
#--use_lora --lora_fused_model_id "models/water_reflection_applied_pix2pix"  --use_fused_conditionmap
# nohup bash train.sh > ./logs/log_train.log 2>&1 &  --perlin_noise  --adaptive_noise --use_lora
# --use_lora --lora_fused_model_id "models/water_reflection_applied_pix2pix"
# nohup bash train2.sh > ./logs/dataset_1301.log 2>&1 &  --use_normalmap --perlin_noise  --adaptive_noise --use_lora
# dataset_1_use_fused_condition_droupout_0.05 nohup bash train.sh > ./logs/dataset_134.log 2>&1 &
# dataset_134_use_fused_condition_droupout_0.05