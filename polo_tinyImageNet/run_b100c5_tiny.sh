python main_tiny.py \
    --fg_nc 100 --task_num 20 --alpha 20 --beta 0.2 --gamma 10 --alpha_2 10 \
    --gpu 0 --cls_weight 0.1 --kd_weight 50 --incre_lr 0.0001 --drift_weight 0.1 --density 1 \
    --log_path log/tiny_b100_c5_polo.txt --file_name polo_b100_c5