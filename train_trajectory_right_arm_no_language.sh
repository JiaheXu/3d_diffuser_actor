main_dir=Actor_Real_right_hand_duck_in_the_bowl

dataset=/ws/3d_diffuser_actor/aloha_data
# valset=/ws/3d_diffuser_actor/aloha_data
valset=/ws/3d_diffuser_actor/aloha_data_eval

lr=1e-4
dense_interpolation=1
interpolation_length=20
num_history=1
diffusion_timesteps=50
B=15
C=120
ngpus=1
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
      main_trajectory.py \
      --tasks duck_in_bowls \
      --dataset $dataset \
      --valset $valset \
      --instructions /ws/3d_diffuser_actor/realworld_demo/realworld_demo/instructions/training.pkl \
      --gripper_loc_bounds /ws/3d_diffuser_actor/aloha_data/14_diffactor_real_tasks_location_bounds.json \
      --gripper_loc_bounds_buffer 0.04 \
      --num_workers 1 \
      --train_iters 100000 \
      --embedding_dim $C \
      --use_instruction 0 \
      --rotation_parametrization 6D \
      --diffusion_timesteps $diffusion_timesteps \
      --val_freq 1000 \
      --dense_interpolation $dense_interpolation \
      --interpolation_length $interpolation_length \
      --exp_log_dir $main_dir \
      --batch_size $B \
      --batch_size_val $B \
      --cache_size 600 \
      --cache_size_val 0 \
      --keypose_only 0 \
      --variations {0..0} \
      --lr $lr\
      --num_history $num_history \
      --cameras front\
      --max_episodes_per_task -1 \
      --quaternion_format $quaternion_format \
      --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-P$prediction_mode-H$num_history-DT$diffusion_timesteps
      # --checkpoint /ws/3dda_models/Aug01/best.pth \
      # --inference 0 \
