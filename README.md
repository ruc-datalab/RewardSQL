# Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards

## :thought_balloon: Introduction

This repository contains the code for our paper "[Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards](https://arxiv.org/pdf/2505.04671)".

RewardSQL enhances Text-to-SQL generation through a comprehensive process-level reward modeling approach. Our framework consists of three interconnected stages:
1. Cold Start with Policy Model and PRM
2. Online RL Training
3. Reward-assistance Inference

![Overview](overview.jpg)

## :open_file_folder: Data Preparation

We provide all necessary datasets in our Google Drive repository.

### Download Datasets
- [RewardSQL-Datasets](https://drive.google.com/file/d/1BKuGOEeuv8V0KGVCfnh195sKtIF2o_Nh/view?usp=drive_link): Contains all training and testing data
  - Bird training data
  - Bird dev data
  - Spider test data

After downloading, extract the datasets:
```sh
mkdir -p data/
unzip RewardSQL-Datasets.zip -d data/
```

The extracted structure should be:
```
data/
├── spider/
│   ├── test.json
│   └── database/
└── bird/
    ├── train.json
    ├── dev.json
    └── database/
```

## :computer: Environment Preparation
![Python](https://img.shields.io/badge/Python-3.11-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.30.0-orange)

Please install the required packages:
```sh
pip install -r requirements.txt
```

Prepare the following folders:
```sh
mkdir -p checkpoints/cocte_model
mkdir -p checkpoints/prm_model
mkdir -p checkpoints/grpo_model
mkdir -p results
```

## :zap: Quick Start

### Download pre-trained models
- [CoCTE SFT Model](https://drive.google.com/file/d/1hP8FO_VA7Lf9wwqHz_Uqvs3ccrSP_x66/view?usp=sharing): Put it under `checkpoints/cocte_model`.
- [Process Reward Model](https://drive.google.com/file/d/1hP8FO_VA7Lf9wwqHz_Uqvs3ccrSP_x66/view?usp=sharing): Put it under `checkpoints/prm_model`.
- [GRPO Trained Model](https://drive.google.com/file/d/1hP8FO_VA7Lf9wwqHz_Uqvs3ccrSP_x66/view?usp=sharing): Put it under `checkpoints/grpo_model`.

### Text-to-SQL inference

Our updated inference process consists of three steps:

1. First, start the LLM service:
```sh
sh evaluation/apply_llm.sh
```

2. Then start the Process Reward Model service:
```sh
CUDA_VISIBLE_DEVICES=0 python evaluation/prm_api.py --port 5050
```

3. Finally, run the evaluation:
```sh
sh evaluation/evaluate_bestofN.sh
```

## :open_hands: Train with GRPO

Our updated training process consists of two steps:

1. First, start the SQL executor service:
```sh
python verl/sql_executor.py
```

2. Then start the GRPO training:
```sh
sh verl/scripts/run_grpo.sh
```

We recommend using `tmux` for managing these different services in separate windows.


## :bar_chart: Results

Our RewardSQL framework achieves outstanding performance on multiple Text-to-SQL benchmarks:

| Model | Bird Dev | Spider Test | 
|-------|------------|------------------|
| Qwen2.5-7B | 52.5 | 75.6 |
| RewardSQL (Greedy) | **59.7** | **77.0** |
| RewardSQL (PRM@32) | **68.9** | **81.7** |

## :speech_balloon: Citation

If our code is helpful to you, please cite our work:
```bibtex
@article{rewardsql2025,
  title={Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards},
  author={Zhang, Yuxin and Fan, Meihao and Fan, Ju and Yi, Mingyang and Luo, Yuyu and Tan, Jian and Li, Guoliang},
  journal={arXiv preprint arXiv:2505.04671},
  year={2025}
}
```
## 🌻 Acknowledgement

We implement our reinforcement learning algorithm extending from [veRL](https://github.com/volcengine/verl) framework. We utilize [vLLM](https://github.com/vllm-project/vllm) for efficient inference and develop evaluation scripts based on [BIRD](https://bird-bench.github.io/) and [Spider](https://yale-lily.github.io/spider) datasets. Thanks for their great contributions!

<!-- ## Release Checklist

- [ ] Models used in the paper
- [ ] Evaluation code
- [ ] Datasets
- [ ] GRPO training code -->