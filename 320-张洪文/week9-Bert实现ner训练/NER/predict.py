import re
from collections import defaultdict
import torch
from model import NERModel
from transformers import BertTokenizer
from loader import load_vocab
from config import Config

class Prediction:
    def __init__(self, config, weights_path):
        self.config = config
        if config["model"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.model = NERModel(config)  # 加载模型
        self.model.load_state_dict(torch.load(weights_path))  # 加载权重

    # 对传入的文本进行ids转换
    def encode_sentence(self, sentence):
        if self.config["model"] == "bert":
            input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_len"],
                                              padding="max_length", truncation=True)
        else:
            input_ids = []
            for char in sentence:
                input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
            # 截断补全
            input_ids = input_ids[: self.config["max_len"]]
            input_ids += [0] * (self.config["max_len"] - len(input_ids))
        return input_ids

    def predict(self, sentence):
        input_ids = self.encode_sentence(sentence)
        input_ids = torch.LongTensor([input_ids])
        self.model.eval()
        with torch.no_grad():
            pred_results = self.model(input_ids)  # 得到预测值
        # 没有使用crf时则直接选择最大的概率值的索引，即类别label
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1).squeeze()
            pred_results = pred_results.tolist()
        else:
            pred_results = pred_results[0]
        labels = "".join([str(x) for x in pred_results])  # 将预测的标签进行拼接
        results = defaultdict(list)  # 保存预测的实体
        # 通过正则匹配模型标注的实体
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


if __name__ == '__main__':
    weight_path = "./models/NER_bert_20.pth"
    # weight_path = "./models/NER_lstm_30.pth"
    pd = Prediction(Config, weight_path)

    que = "在2023年8月20日下午，上海徐汇区的国际文化交流中心内，华艺画廊举办了一场艺术展览。开幕式上，创始人张晓明向参观者介绍了展览的独特之处"
    que = "被中国男队称为“三只虎”的欧洲名将萨姆索诺夫、普里莫拉茨、瓦尔德内尔这次没有报名参赛,但上个月刚夺得欧洲锦标赛男团冠军的法国队派出了全部主力,盖亭、希拉、勒古、埃洛瓦都将参加角逐。"
    # que = "正是在这里,1984年10月1日,我国改革开放的总设计师邓小平同志在广场检阅三军官兵,并登上天安门城楼,向参加国庆35周年庆典的群众队伍发表讲话,他强调:“当前的主要任务,是要对妨碍我们前进的现行经济体制,进行有系统的改革。"
    result = pd.predict(que)
    print("提取的实体信息如下:")
    for k, v in result.items():
        print(f"\t{k}: {v}")
    # while True:
    #     question = input("请输入问题：")
    #     res = qas.query(question)
    #     print("命中问题：", res)
    #     print("-----------")

