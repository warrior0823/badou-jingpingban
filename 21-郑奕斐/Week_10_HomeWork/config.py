import os

Config = {
    "model_path": "output",
    'bert_path': r"/Users/henryzheng/Desktop/NLP/八斗学院/bert-base-chinese",
    "corpus_path": "corpus.txt",
    "vocab_path":"vocab.txt",
    'train_sample' : 10000,
    'window_size' : 10,
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 3e-6,
}