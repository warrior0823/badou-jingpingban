import torch

"""
模型配置
"""
Config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese",
    "bert_vocab_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese\vocab.txt",
    "vocab_path": "./data/chars.txt",  # 以字作为字符集
    "schema_path": "./data/schema.json",
    "train_path": "./data/train",
    "valid_path": "./data/test",
    "model_save_path": "./models/",
    "model": "bert",
    "save_model": True,
    "seed": None,   # 是否设置随机种子
    "max_len": 80,
    "epochs": 10,
    "batch_size": 16,
    # "embedding_size": 512,  # lstm 512
    # "hidden_size": 1024,  # lstm 1024
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,   # 是否使用 CRF 条件随机场

    "tuning_tactics": "lora_tuning",   # peft的策略
    "num_labels": 9,   # 实体类别数
    "inference_mode": False
}
