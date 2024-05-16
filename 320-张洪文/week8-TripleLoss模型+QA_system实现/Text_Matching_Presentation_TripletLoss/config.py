import torch

"""
模型配置
"""
Config = {
    # "vocab_path": "../data/words.txt",  # 以词作为字符集
    "vocab_path": r"E:\个人学习\人工智能\NLP_Code\7.文本匹配\data\chars.txt",    # 以字作为字符集
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_path": r"E:\个人学习\人工智能\NLP_Code\7.文本匹配\data\train.json",
    "valid_path": r"E:\个人学习\人工智能\NLP_Code\7.文本匹配\data\valid.json",
    "schema_path": r"E:\个人学习\人工智能\NLP_Code\7.文本匹配\data\schema.json",
    "model_save_path": "./",
    "save_model": True,
    "samples_number": 500,  # 每轮训练的随机采样样本数
    "positive_sample_rate": 0.5,  # 正样本采样比例
    "max_len": 20,
    "epochs": 30,
    "batch_size": 128,
    "embedding_size": 256,
    "hidden_size": 512,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": None,
}
