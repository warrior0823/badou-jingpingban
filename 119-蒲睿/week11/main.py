from transformers import BertTokenizer, BertModel
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import LLM_Module
# from evaluate import Evaluator
from loader import load_data
from generator import generate_sentence
from loader import build_dataset
import transformers


#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    train_data = load_data(config["train_data_path"])
    
    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
    train_data_set = build_dataset(tokenizer, train_data, config["max_length"], config["batch_size"])
    
    model = LLM_Module(config)
    
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        logger.info("Epoch %d begin" % epoch)
        train_loss = []
        for x, mask, y in train_data_set:
            if cuda_flag:
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            
            optimizer.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("Average Batch loss %f" % (np.mean(train_loss)))
        print(generate_sentence("文化", model, tokenizer))
        print(generate_sentence("科技", model, tokenizer))
        
if __name__ == '__main__':
    main(Config)
