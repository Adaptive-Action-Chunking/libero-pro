#!/bin/bash

cd /home/sangfor/code/lyc/LIBERO-PRO/pi05_benhmark


# python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object_temp_x0.2 --port 8000 --out_path ~/Videos/libero_pro_pi05/action_entropy --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 3 --seed 42
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object_temp_x0.2 --port 8000 --out_path ~/Videos/libero_pro_pi05/action_entropy --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 3 --seed 43
python inference_client_action_entropy_pi05.py --num_trials_per_task 50 --task_suite_name libero_object_temp_x0.2 --port 8000 --out_path ~/Videos/libero_pro_pi05/action_entropy --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --move_th 3 --seed 44
