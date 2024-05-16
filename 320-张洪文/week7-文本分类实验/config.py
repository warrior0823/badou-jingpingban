import torch

"""
配置参数信息
"""
Config = {
    "bert_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "model/",
    "data_path": "../data/文本分类练习.csv",
    "vocab_path": "../data/chars.txt",
    "model": "cnn",
    "pooling_type": "max",
    "optimizer": "adm",
    "epochs": 5,
    "batch_size": 512,
    "max_len": 30,   # 文本最大长度
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "learning_rate": 1e-3,
    "seed": 987,
    "pred_number": 1000,
    "class_num": 1,
}
