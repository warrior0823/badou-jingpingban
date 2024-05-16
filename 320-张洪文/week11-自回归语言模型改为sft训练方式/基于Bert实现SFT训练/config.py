import torch

"""
模型配置
"""
Config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese",
    "bert_vocab_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese\vocab.txt",
    "vocab_path": "./data/vocab.txt",  # 以字作为字符集
    "train_path": "./data/sample_data.json",
    "valid_path": "./data/sample_data.json",
    "model_save_path": "./models/",
    "save_model": False,
    "seed": None,   # 是否设置随机种子
    "max_len": 50,
    "epochs": 3000,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-5,
}
