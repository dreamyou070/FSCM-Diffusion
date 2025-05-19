# 백그라운드 실행 예시:
# nohup bash train_timing.sh > ./logs/train_flood_1_neg_cond_0.5.log 2>&1 &
export MODEL_NAME="timbrooks/instruct-pix2pix"
export ACCELERATE_FIND_UNUSED_PARAMETERS=true
export main_process_port=53272
config_file="acc_1.yaml"
disaster="flood"
export DATASET_ID="ParkSY/FSCM_Flood"
file_name="1_only_labeling_unconditional_prob_0.5"

accelerate launch \
  --config_file ../accelerate_config/$config_file \
  --main_process_port $main_process_port \
  --mixed_precision="fp16" \
  train_timing.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_ID \
  --base_label_number 0 \
  --num_levels 5 \
  --resolution=512 \
  --random_flip \
  --train_batch_size=7 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=150000 \
  --learning_rate=5e-06 \
  --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --wandb_project_name="cctv_project" \
  --wandb_name="flood_only_labeling_unconditional_prob_0.5" \
  --output_dir="../result/$disaster/$file_name" \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=50 \
  --seed 42 \
  --conditioning_dropout_prob=0.05