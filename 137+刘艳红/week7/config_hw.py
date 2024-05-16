# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": "文本分类练习.csv",
    # "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path":r'E:\Pycharm_learn\pythonProject1\wk7\nn_pipline\chars.txt',
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\Pycharm_learn\pythonProject1\wk6\bert-base-chinese",
    "seed": 987
}

