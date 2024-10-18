# MODDP: A Multi-modal Open-domain Chinese Dataset for Dialogue Discourse Parsing
Implementation of the paper ```MODDP: A Multi-modal Open-domain Chinese Dataset for Dialogue Discourse Parsing```. The paper has been accepted in Findings ACL 2024.

## Abstract
Dialogue discourse parsing (DDP) aims to capture the relations between utterances in the dialogue. In everyday real-world scenarios, dialogues are typically multi-modal and cover open-domain topics. However, most existing widely used benchmark datasets for DDP contain only textual modality and are domain-specific. This makes it challenging to accurately and comprehensively understand the dialogue without multi-modal clues, and prevents them from capturing the discourse structures of the more prevalent daily conversations. This paper proposes MODDP, the first multi-modal Chinese discourse parsing dataset derived from open-domain daily dialogues, consisting 864 dialogues and 18,114 utterances, accompanied by 12.7 hours of video clips. We present a simple yet effective benchmark approach for multi-modal DDP. Through extensive experiments, we present several benchmark results based on MODDP. The significant improvement in performance from introducing multi-modalities into the original textual unimodal DDP model demonstrates the necessity of integrating multi-modalities into DDP.
## Requirements

Pytorch >= 2.1.1

Transformers >= 4.18.0

## Data Preparation
You can directly load the text data from the `dataset` folder and download the image and audio features from [all_features.pkl](https://pan.quark.cn/s/652af8a14776).

If the link is broken or you need the original video data, please contact gongchen18@suda.edu.cn.

## Training
```bash
python main.py \
    --config_file ./config.cfg \
    --seed 42 \
    --postfix experiments/train \
    --text_plm_name_or_path /path/to/roberta \
    --vision_plm_name_or_path /path/to/vit \
    --audio_plm_name_or_path /path/to/wav2vec2 \
    --bert_path /path/to/bert \
```
Or run directly
```bash
bash run.sh
```

## Predict and Evaluation
```bash
python main.py \
    --config_file ./config.cfg \
    --seed 42 \
    --postfix experiments/predict \
    --text_plm_name_or_path /path/to/roberta \
    --vision_plm_name_or_path /path/to/vit \
    --audio_plm_name_or_path /path/to/wav2vec2 \
    --bert_path /path/to/bert \
    --ckpt_path /path/to/best/model \
    --train False \
    --predict True \
```

## Citation
If you find this repo helpful, please cite the following paper: 
```@inproceedings{gong2024moddp,
  title={MODDP: A Multi-modal Open-domain Chinese Dataset for Dialogue Discourse Parsing},
  author={Gong, Chen and Kong, Dexin and Zhao, Suxian and Li, Xingyu and Fu, Guohong},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  year={2024}
}
```
