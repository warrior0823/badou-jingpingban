import torch

"""
模型配置
"""
Config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese",
    "vocab_path": "./data/chars.txt",  # 以字作为字符集
    "schema_path": "./data/schema.json",
    "train_path": "./data/train",
    "valid_path": "./data/test",
    "model_save_path": "./models/",
    "save_model": True,
    "model": "bert",
    "seed": None,   # 是否设置随机种子
    "class_num": 9,
    "max_len": 100,
    "num_layers": 2,
    "epochs": 20,
    "batch_size": 8,
    "embedding_size": 512,  # lstm 512
    "hidden_size": 1024,  # lstm 1024
    "optimizer": "adam",
    "learning_rate": 5e-6,
    "use_crf": False,   # 是否使用 CRF 条件随机场
}
