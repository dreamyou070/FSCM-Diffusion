# 백그라운드 실행 예시:

# nohup bash train_stylematching.sh > ./logs/train_fused_no_distill.log 2>&1 &
export MODEL_NAME="timbrooks/instruct-pix2pix"
export ACCELERATE_FIND_UNUSED_PARAMETERS=true
export main_process_port=52343
config_file="acc_3.yaml"
disaster="flood"
export DATASET_ID="ParkSY/FSCM_Flood"
file_name="6_fscm_teacher_distill"
accelerate launch \
  --config_file ../accelerate_config/$config_file \
  --main_process_port $main_process_port \
  --mixed_precision="fp16" \
  train_stylematching.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_ID \
  --use_normalmap \
  --use_depthmap \
  --base_label_number 0 \
  --num_levels 5 \
  --conditioning_dropout_prob=0.05 \
  --resolution=512 \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=150000 \
  --learning_rate=5e-05 \
  --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --wandb_project_name="cctv_project" \
  --wandb_name="flood_fscm_teacher_distill" \
  --output_dir="../result/$disaster/$file_name" \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=50 \
  --seed 42 \
  --use_fused_conditionmap \
  --conditioning_dropout_prob=0.05 \
  --style_loss_lambda 0.0 \
  --lora_fused_model_id "./pretrained_model/water_reflection_applied_pix2pix2"

# 'models/water_reflection_applied_pix2pix'
# 'models/snow_applied_pix2pix'
# --conditioning_dropout_prob=0.05
#--use_lora --lora_fused_model_id "models/water_reflection_applied_pix2pix"
# nohup bash train_stylematching.sh > ./logs/log_train_stylematching.log 2>&1 &  --use_normalmap --perlin_noise  --adaptive_noise --use_lora