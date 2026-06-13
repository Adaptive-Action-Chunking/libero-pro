#!/bin/bash

cd /home/sangfor/code/lyc/LIBERO-PRO/

python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.2 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 42
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.3 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 42
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.4 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 42

python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.2 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 43
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.3 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 43
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.4 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 43

python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.2 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 44
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.3 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 44
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.4 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 44