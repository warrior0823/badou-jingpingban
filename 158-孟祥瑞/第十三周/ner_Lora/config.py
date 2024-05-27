# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "tuning_tactics": "lora_tuning",
    "optimizer": "adam",
    "pooling_style": "max",
    "learning_rate": 1e-4,
    "class_num": 9,
    "use_crf": False,
    "seed": 987,
    "pretrain_model_path": r"C:\Users\28194\PycharmProjects\pythonProject2\week6\week6 语言模型和预训练\下午\bert-base-chinese",
}
