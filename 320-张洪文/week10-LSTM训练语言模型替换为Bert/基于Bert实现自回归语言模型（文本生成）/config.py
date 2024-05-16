import torch

"""
模型配置
"""
Config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bert_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese",
    "bert_vocab_path": r"E:\个人学习\人工智能\NLP_Code\pretrain_models\bert-base-chinese\vocab.txt",
    "vocab_path": "./data/vocab.txt",  # 以字作为字符集
    "train_path": "./data/corpus.txt",
    "valid_path": "./data/corpus.txt",
    "model_save_path": "./models/",
    "save_model": True,
    "model": "bert",
    "seed": None,   # 是否设置随机种子
    "train_sample_number": 50000,
    "test_sample_number": 1000,
    "window_size": 20,
    "epochs": 20,
    "batch_size": 512,
    "embedding_size": 256,
    "hidden_size": 512,
    "optimizer": "adam",
    "learning_rate": 1e-5,
}
