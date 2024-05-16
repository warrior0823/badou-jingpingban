from torch.utils.data import DataLoader
import json
import torch
from transformers import BertTokenizer

def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)

class DataGenerator:
    def __init__(self,data_path,config,logger) -> None:
        self.config = config
        self.logger = logger
        self.path = data_path
        self.tokenizer = load_vocab(config["vocab_path"])
        self.load()
    
    #sft的数据构造
    #loss只计算答案部分，通过mask矩阵，让上下文之间没有交互
    #label中使用-1，表示不参与训练
    def load(self):
        self.data = []
        with open(self.path,"r",encoding="utf8") as f:
            for i ,line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                x,mask,y = self.prepare_date(title,content)
            self.data.append([x,mask,y])
        return 

    def prepare_date(self,prompt,answer):
        max_length= self.config["max_length"]
        prompt_encode = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = self.tokenizer.encode(answer, add_special_tokens=False)
        # sft 的输入数据x 是[cls]+prompt+            [sep]+[answer]+[sep]
        # y 是                  [-1]*len(prompt) +  [-1] +[answer]+[-1]
        #  mask 矩阵的形状为： len(prompt)+2+len(answer)+1,len(prompt)+2+len(answer)+1
        # mask矩阵  len(prompt)+2 行的 len(prompt)+2： 均为0
         # mask矩阵 len(prompt)+2 + i 行的 len(prompt)+2 +i+1： 均为0
        x = [self.tokenizer.cls_token_id] + prompt_encode + [self.tokenizer.sep_token_id] + answer_encode + [self.tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [self.tokenizer.sep_token_id] + [-1]
        #构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = self.create_mask(len(prompt_encode), len(answer_encode))
        #padding x,y的padding均为0
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        #  mask pad 如果原始mask 已经max_length 大了，会截取。 如果比原来的小会填充0
        mask = self.pad_mask(mask, (max_length, max_length))
        return x,mask,y

    
    #构造掩码，输入两个字符串的长度
    def create_mask(self,s1, s2):
        len_s1 = s1 + 2 #cls + sep
        len_s2 = s2 + 1 #sep
        # 创建掩码张量
        mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
        # 遍历s1的每个token
        for i in range(len_s1):
            # s1的当前token不能看到s2的任何token
            mask[i, len_s1:] = 0  
        # 遍历s2的每个token
        for i in range(len_s2):
            # s2的当前token不能看到后面的s2 token
            mask[len_s1 + i, len_s1 + i + 1:] = 0
        return mask

    def pad_mask(self,tensor, target_shape):
        # 获取输入张量和目标形状的长宽
        height, width = tensor.shape
        target_height, target_width = target_shape
        # 创建一个全零张量,形状为目标形状
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        # 计算需要填充或截断的区域
        h_start = 0
        w_start = 0
        h_end = min(height, target_height)
        w_end = min(width, target_width)
        # 将原始张量对应的部分填充到全零张量中
        result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
        return result
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



def load_data(data_path,config,logger,shuffle=True):
    dg = DataGenerator(data_path,config,logger)
    # num_workers = 0
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl