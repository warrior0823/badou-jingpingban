# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "schema_path": "data/schema.json",
    "train_data_path": "data/train",
    "valid_data_path": "data/test",
    "vocab_path":"vocab.txt",
    "tuning_tactics":"lora_tuning",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"/Users/AllenXie/Documents/AIVideo/nlp/bert-base-chinese",
    "pretrain_model_path": r"/Users/AllenXie/Documents/AIVideo/nlp/bert-base-chinese",
    "vocab_size": 21128,
    "model_path": "model",
    "seed": 987
}