# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (LIBER-PRO Evaluation)

This is the official codebase of AAC.

[**[Home page]**](https://lance-lot.github.io/adaptive-chunking.github.io/) &ensp; [**[Paper]**](https://arxiv.org/abs/2604.04161)

-------
## Overview
For a quick review of the implentation of AAC algorithm, see the function select_chunk_size in [action_entropy_v2.py](https://github.com/Adaptive-Action-Chunking/libero-pro/blob/main/action_optimization/action_entropy_v2.py).

This repo is built on offcial [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) with modifications to support AAC. If you already have LIBERO-PRO installed, copy and place the folders (1) action_optimization (2) gr00t_benchmark (3) pi05_benchmark from this repo to your original LIBERO-PRO projects. If you don't have LIBERO-PRO installed before, follow installation instructions from official [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO). Then add in the above mentioned 3 folders from this repo.

Note:
If you want to align the LIBERO-PRO version with our experiments, you can use the command below to specify the LIBERO-PRO codebase version.

In the project folder of installed LIBERO-PRO:
```sh
git checkout 13613f0cc90e3869b157d34f7a88e7d6041cd423
```


## Usage
Our implementation is based on server-client mode. This repo only includes the client part. To run a policy server, we provide the implementation of [GR00T N1.5](https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample) (with modification to support sampling mutiple action chunks in parallel.)

example to run the evaluation client for one task with AAC:
```sh
cd gr00t_benchmark
python gr00t_benchmark/inference_client_gr00t.py  --task_suite_name libero_object_temp_x0.2 --port 8091 --out_path ~/Videos/libero_pro/gr00t --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --seed 42
```

Configure --task_suite_name for different task suites in LIBERO-PRO(The suffix x0.2 indicates the level of OOD is 0.2 along x axis.), set --port according to your policy server.

Refer to [gr00t_benchmark/task_schedule.sh](https://github.com/Adaptive-Action-Chunking/libero-pro/blob/main/gr00t_benchmark/task_schedule.sh) for more example scripts.
 
-------
## Citation
```bibtex
@inproceedings{liang2026adaptive,
  title={Adaptive action chunking at inference-time for vision-language-action models},
  author={Liang, Yuanchang and Wang, Xiaobo and Wang, Kai and Wang, Shuo and Peng, Xiaojiang and Chen, Haoyu and Chua, David Kim Huat and Vadakkepat, Prahlad},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20802--20811},
  year={2026}
}
```

