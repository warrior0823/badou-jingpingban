import logging
import random
import numpy as np
import torch
import os


from config import Config
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator

from peft import LoraConfig,TaskType,PromptEncoderConfig,PromptTuningConfig,PrefixTuningConfig,get_peft_model

logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型主训练流程
"""
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"],config)

    model = TorchModel

    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics =="lora_tuning":
        peft_config = LoraConfig(
            task_type = TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules = ["query","key","value"]
        )
    elif tuning_tactics =="p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS",num_virtual_tokens = 10)
    elif tuning_tactics == "prompt_tuning":
        peft_config=PromptTuningConfig(task_type="SEQ_CLS",num_virtual_tokens=10)
    elif tuning_tactics =="prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS",num_virtual_tokens=10)

    model = get_peft_model(model,peft_config)

    cuda_flag = torch.cuda.is_available()
    # cuda_flag = False
    if cuda_flag:
        logger.info("gpu可以使用，迁移到gpu")
        model = model.cuda()
    
    optim = choose_optimizer(config,model)

    evaluator = Evaluator(config,model,logger)

    for epoch in range(config["epoch"]):
        model.train()
        train_loss = []
        for index,batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            # 归零
            optim.zero_grad()
            input_ids,labels = batch_data
            # print(f"input_ids:{input_ids.shape},labels:{labels.shape}")
            output = model(input_ids)[0]
            # print(f"output  shape :{output.shape},labels:{labels.shape}")

            loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(output.view(-1,output.shape[-1]),labels.view(-1))
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            train_loss.append(loss.item())
            if index % int(len(train_data)/2) ==0:
                # 一半的训练数据记录下loss
                logger.info(f"epoch:{epoch+1},train_loss:{loss.item()}")
        
        logger.info(f"epoch average loss {np.mean(train_loss)}")
        acc=evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"],f"epoch_{epoch+1}.pth")
    save_tunable_parameters(model,model_path)
    return acc

def save_tunable_parameters(model,model_path):
    saved_parameters={
        k: v.to("cpu") for k,v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_parameters,model_path)
    
if __name__ == "__main__":
    main(Config)

