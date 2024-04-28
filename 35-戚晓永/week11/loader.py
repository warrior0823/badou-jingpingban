import numpy as np
from torch.utils.data import DataLoader

from config import config
import json


# 加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus():
    with open(config["train_data_path"], "r", encoding='utf-8') as f:
        data = []
        for line in f:
            # json.load(file)接收一个文件对象，其内部为json格式，json.loads(line)接收一个json格式的字符串
            line = json.loads(line)
            data.append([line['title'], line['content']])
    return data


# 构建训练集合，内部单元的格式为[x,y,mask]
def build_dataset(tokenizer, max_length=512):
    '''
    全局定义打印Numpy数时的行为
    threshold：指定NumPy在打印数组时应该展示多少维度的数组。默认值是1000。threshold=np.inf意味着NumPy将打印数组的任何维度，无论其大小。这通常用于打印大型数组，以确保不会因为元素数量太多而省略任何内容。
    linewidth：指定每行打印的字符数。默认值是75。如果数组内容太长，它会被自动折行打印，以适应指定的行宽。这有助于更清晰地查看数组内容。
    precision：指定浮点数的小数点精度。默认值是8。例如一个浮点数123.456789，precision=2将打印为123.45。
    '''
    np.set_printoptions(threshold=100000, linewidth=1000, precision=5)

    data = load_corpus()
    data = np.array(data)

    data_set = []
    for title, content in data:
        # x shape [batch_size,title_length]
        title = tokenizer.encode(title, add_special_tokens=False)
        # x shape [batch_size,content_length]
        content = tokenizer.encode(content, add_special_tokens=False)
        # 构建 seq to seq 的mask,形状为[batch_size,sentence_length]
        # x_i 的内容为       cls, x1, x2,  xn,  seq, y1, yn,seq
        # y_i 的内容为        x1, x2, xn, seq,  y1, yn, seq,-1
        # y_label_i 的内容为 -1,  -1, -1,  -1,  y1, yn, seq,-1
        # x形状[batch_size,cls+title_length+sep+content_length+sep]
        x = [tokenizer.cls_token_id] + title + [tokenizer.sep_token_id] + content + [tokenizer.sep_token_id]
        y = len(title) * [-1] + [-1] + content + [tokenizer.sep_token_id] + [-1]
        # 生成掩码
        mask = np.zeros((max_length, max_length))
        mask[:len(x), :len(x) - len(content) - 1] = 1
        mask_y = np.tril(np.ones((len(content) + 1, len(content) + 1)))
        mask[1 + len(title) + 1:len(x), 1 + len(title) + 1:len(x)] = mask_y
        x = x + (max_length - len(x)) * [tokenizer.pad_token_id]
        y = y + (max_length - len(y)) * [tokenizer.pad_token_id]
        data_set.append([np.array(x), np.array(y), mask])
        # for data in data_set:
        #     x = ''
        #     y = ''
        #     for i in range(len(data[0])):
        #         x += tokenizer.decode(data[0][i])
        #         y += tokenizer.decode(data[1][i])
        #     print(x)
        #     print(y)
        #     print(data[2])

    return DataLoader(data_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
