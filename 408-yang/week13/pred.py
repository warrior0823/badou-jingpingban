import logging


from config import Config
from model import TorchModel
from evaluate import Evaluator
import os

import torch
from peft import get_peft_model,LoraConfig,TaskType,PromptEncoderConfig,PrefixTuningConfig,PromptTuningConfig

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tuning_tactics = Config["tuning_tactics"]
print(f"正在使用{tuning_tactics}")

if tuning_tactics =="lora_tuning":
    peft_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        inference_mode=True,
        r = 8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query","key","value"]
    )

elif tuning_tactics =="p_tuning":
    peft_config= PromptEncoderConfig(
        task_type="SEQ_CLS",num_virtual_tokens=10
    )
elif tuning_tactics =="prompt_tuning":
    peft_config=PromptTuningConfig(
        task_type="SEQ_CLS",num_virtual_tokens=10
    )
elif tuning_tactics =="prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS",num_virtual_tokens=10)

model = TorchModel

model = get_peft_model(model,peft_config=peft_config)

state_dict = model.state_dict()
print(state_dict.keys())
model_lora_path = os.path.join(Config["model_path"],"epoch_10.pth")
if tuning_tactics =="lora_tuning":
    state_dict.update(torch.load(model_lora_path))

print(f"\nstate_dict:{state_dict.keys()}")
model.load_state_dict(state_dict)

for k,v in model.named_parameters():
    if v.requires_grad:
        print(k,v.shape)

model = model.cuda()
evalutor = Evaluator(Config,model,logger)
evalutor.eval(0)