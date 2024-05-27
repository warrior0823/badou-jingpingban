

'''
参数信息配置

'''

objConfig = {
    "model_path": "model_output",
    "schema_path" : "data/schema.json",
    "train_data_path" : "data/train.json",
    "valid_data_path" : "data/valid.json",
    "model_type" : "cnn",
    "vocab_path" : "data/chars.txt",
    "max_len" : 20 ,            #文本最大长度
    "hidden_size" : 128 ,       #隐藏层
    "epoch": 20 ,               #轮数
    "batch_size" : 32,          #每轮训练的批次的大小
    "epoch_data_size" : 200,    #每轮训练的采样数
    "positive_sample_rate" : 0.5 , #正样本比例
    "optimizer" : "adam" ,          #优化器
    "learning_rate" : 1e-3 ,        #学习率

}