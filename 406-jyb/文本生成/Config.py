import torch
class CFG:
    def __init__(self):
        self.vocab_path=r"E:\python\pre_train_model\bert_file\vocab.txt"
        self.pre_model_path=r"E:\python\pre_train_model\bert_file"
        self.lr=0.001
        self.epoch=10
        self.data_path=r"E:\badouFile\第十周\week10 文本生成问题\week10 文本生成问题\lstm语言模型生成文本\corpus.txt"
        self.cuda=False
        self.window_size=20
        self.max_char=self.window_size+1
        self.hidden_size=768
        self.batch_size=64
