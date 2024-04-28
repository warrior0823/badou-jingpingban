import random
import torch
from transformers import BertTokenizer
from config import Config

"""
数据加载格式：
    train_dataset：train_number [sen_encode, label]
    test_dataset： test_number [sen_encode, label]
    pred_dataset： pred_number [sen_encode, label]
"""
# 加载词汇集
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            # 0 是留给padding的位置，所以从1开始。在模型结构文件中可见配置
            token_dict[token] = index+1
    return token_dict

# 文本编码
def encode_sentence(sentence, vocab, config):
    sentence_ids = []
    # 获取每个字符对应的标签
    for word in sentence:
        sentence_ids.append(vocab.get(word, vocab.get('[UNK]')))
    # 按照最大长度进行补全、截断,使其可以在同一batch内运算
    if len(sentence_ids) < config["max_len"]:
        sentence_ids.extend([0] * (config["max_len"]-len(sentence_ids)))
    else:
        sentence_ids = sentence_ids[: config["max_len"]]
    return sentence_ids

# 数据集处理
def sentence_dispose(data, vocab, config):
    x = []
    y = []
    for line in data:
        label, review = line.strip().split(",", maxsplit=1)
        # bert类模型，需要使用bert的tokenizer对文本进行编码
        if "bert" in config["model"]:
            tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
            # pad_to_max_length=True 指定编码后的序列长度小于max_len，那么它应该被填充padded至max_len
            input_text_encode = tokenizer.encode(review, max_length=config["max_len"], padding="max_length", truncation=True)
        else:
            input_text_encode = encode_sentence(review, vocab, config)  # 使用自定义编码函数
        # 类型转换
        x.append(input_text_encode)
        y.append(int(label))
    return [torch.LongTensor(x), torch.FloatTensor(y)]

# 数据加载
def data_load(config):
    vocab = load_vocab(config["vocab_path"])  # 加载词汇表
    config["vocab_size"] = len(vocab)
    # 读取文件
    with open(config["data_path"], 'r', encoding='utf-8') as f:
        text = f.readlines()[1:]  # 第一行为标题
    random.shuffle(text)  # 打乱列表顺序

    # 先取出预测集的数据
    pred_data = []
    for i in range(config["pred_number"]):
        element = random.choice(text)  # 从剩余列表中随机选择一个元素
        pred_data.append(element)  # 将选中的元素添加到结果列表中
        text.remove(element)  # 从原列表中移除选中的元素
    # 再取出训练集和测试集的数据
    train_num = int(len(text) * 0.8)
    train_data = text[: train_num]  # 训练集数据为总数据的80%
    test_data = text[train_num:]    # 测试集数据为总数据的20%

    # 对数据集数据进行处理，得到符合模型输入的数据
    pred_data = sentence_dispose(pred_data, vocab, config)
    train_data = sentence_dispose(train_data, vocab, config)
    test_data = sentence_dispose(test_data, vocab, config)

    return train_data, test_data, pred_data


# 测试数据生成
if __name__ == '__main__':
    t_data, v_data, p_data = data_load(Config)
