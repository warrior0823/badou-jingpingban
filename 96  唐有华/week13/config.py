# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "train_data_path": "data/train_tag_news.json",
    "valid_data_path": "data/valid_tag_news.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 20,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "tuning_tactics": "lora_tuning",
    "learning_rate": 1e-4,
    "bert_path": r"D:\git\open_object\bert-base-chinese"
}