import re
from collections import defaultdict
import torch
from peft import LoraConfig, TaskType, PromptEncoderConfig, PromptTuningConfig, PrefixTuningConfig, get_peft_model

from model import NERModel, bert_model
from transformers import BertTokenizer, AutoModelForTokenClassification
from loader import load_vocab
from config import Config


def load_peft_model(peft_model_path):
    tuning_tactics = Config["tuning_tactics"]
    print("peft训练策略：", tuning_tactics)
    Config["inference_mode"] = True
    model = NERModel(Config)
    state_dict = model.state_dict()

    # 将微调部分权重加载进模型
    peft_state_dict = torch.load(peft_model_path)
    # for k, v in peft_state_dict.items():
    #     print(k, v.shape)

    state_dict.update(peft_state_dict)
    # 加载微调后的模型
    model.load_state_dict(state_dict)
    return model


class Prediction:
    def __init__(self, config, weights_path):
        self.config = config
        if config["model"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.model = load_peft_model(weights_path)  # 加载模型
        # self.model.load_state_dict(torch.load(weights_path))  # 加载权重

    # 对传入的文本进行ids转换
    def encode_sentence(self, sentence):
        if self.config["model"] == "bert":
            input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_len"],
                                              padding="max_length", truncation=True,
                                              add_special_tokens=True)
        else:
            input_ids = []
            for char in sentence:
                input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
            # 截断补全
            input_ids = input_ids[: self.config["max_len"]]
            input_ids += [0] * (self.config["max_len"] - len(input_ids))
        return input_ids

    def predict(self, sentence):
        input_ids = self.encode_sentence(list(sentence))
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
        labels = "".join([str(x) for x in pred_results[1:-1]])  # 将预测的标签进行拼接
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
    weight_path = "./models/NER_bert_10_0.001_87%.pth"
    # weight_path = "./models/NER_lstm_30.pth"
    pd = Prediction(Config, weight_path)

    que = "在2023年8月20日下午，上海徐汇区的国际文化交流中心内，华艺画廊举办了一场艺术展览。开幕式上，创始人张晓明向参观者介绍了展览的独特之处"
    # que = "被中国男队称为“三只虎”的欧洲名将萨姆索诺夫、普里莫拉茨、瓦尔德内尔这次没有报名参赛,但上个月刚夺得欧洲锦标赛男团冠军的法国队派出了全部主力,盖亭、希拉、勒古、埃洛瓦都将参加角逐。"
    result = pd.predict(que)
    print("提取的实体信息如下:")
    for k, v in result.items():
        print(f"\t{k}: {v}")
    # while True:
    #     question = input("请输入问题：")
    #     res = qas.query(question)
    #     print("命中问题：", res)
    #     print("-----------")

