config ={
    "model_path": "./model/",
    # "input_max_length": 120,
    # "output_max_length": 30,
    "epoch": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate":1e-3,
    "seed":42,
    "vocab_size":6219,
    "vocab_path": r"D:/aa-aziliao/badou/pretrain/bert_base_chinese/vocab.txt",
    # r前缀确保了字符串"./data/sample_data.json"被解释为它表面的样子，而不是将反斜杠\作为转义字符
    # 。这在处理文件路径时特别有用，因为它允许你写出更直观的路径，而不需要担心转义序列。
    "train_data_path": r"./data/sample_data.json",
    "valid_data_path": r"./data/sample_data.json",
    "pretrain_model_path": r"D:/aa-aziliao/badou/pretrain/bert_base_chinese",
    "beam_size":5,
    "max_length":50
    }