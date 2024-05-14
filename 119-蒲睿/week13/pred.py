import logging
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from config import Config
from transformers import BertTokenizer
from peft import LoraConfig, PromptEncoderConfig, PromptTuningConfig, PrefixTuningConfig, get_peft_model, TaskType
from model import TorchModel_auto
from evaluator import Evaluator
import re

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tuning_strategy = Config["tuning_strategy"]
if tuning_strategy == "lora":
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
elif tuning_strategy == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10) # 虚拟token的数量
elif tuning_strategy == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
elif tuning_strategy == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    
model = TorchModel_auto
model = get_peft_model(model, peft_config)
state_dict = model.state_dict()
state_dict.update(torch.load('model_output/epoch_20.pth'))

# if tuning_strategy == "lora_tuning":
#     state_dict.update(torch.load('model_output/lora_tuning.pth'))
# elif tuning_strategy == "p_tuning":
#     state_dict.update(torch.load('model_output/p_tuning.pth'))
# elif tuning_strategy == "prompt_tuning":
#     state_dict.update(torch.load('model_output/prompt_tuning.pth'))
# elif tuning_strategy == "prefix_tuning":
#     state_dict.update(torch.load('model_output/prefix_tuning.pth'))

model.load_state_dict(state_dict)
sentence = "2024年5月8日，在梅州的梅大高速上发生重大交通事故，梅州市交警部门迅速展开调查，事故原因正在进一步调查中。"
tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])
input_id = torch.LongTensor([tokenizer.encode(sentence, max_length=Config["max_length"], padding=True, truncation=True)])
predict = model(input_id)[0]
pred_label = torch.argmax(predict, dim=-1).tolist()[0]

mapping = {'名字':[],'时间':[],'地点':[],'机构':[]}
def decode(labels, sentence):
    labels = "".join(str(x) for x in labels)
    for location in re.finditer('(26+)', labels):
        s,e = location.span()
        mapping['名字'].append(sentence[s:e])
    for location in re.finditer('(37+)', labels):
        s,e = location.span()
        mapping['时间'].append(sentence[s:e])
    for location in re.finditer('(15+)', labels):
        s,e = location.span()
        mapping['机构'].append(sentence[s:e])
    for location in re.finditer('(04+)', labels):
        s,e = location.span()
        mapping['地点'].append(sentence[s:e])
decode(pred_label, sentence)
for k, v in mapping.items():
    print(k,v)


