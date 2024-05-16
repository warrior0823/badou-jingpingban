# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "model_type": "bert",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 12,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 5e-5,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"C:\Users\ADMIN\Desktop\NLP\bert\bert-base-chinese"
}
