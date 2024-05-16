config = {
    "model_path": "./model",
    "schema_path": "./data/ner/schema.json",
    "train_data_path": "./data/ner/train.txt",
    "valid_data_path": "./data/ner/test.txt",
    "vocab_path":"./data/chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3, 
    "need_cut": 1,
    "class_num":9,
    "num_layers":2,
    "use_crf": True,
    "pretrain_model_path": r"/root/pretrain/bert_base_chinese"
}