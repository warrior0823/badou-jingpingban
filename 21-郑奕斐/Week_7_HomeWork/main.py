# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import time
import pandas as pd
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model_data = []

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    #创建保存模型参数目录
    model_params_dir = 'model_params'
    if not os.path.isdir(model_params_dir):
        os.mkdir(model_params_dir)
    
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用, 迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    start_time = time.time()
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))

        acc = evaluator.eval(epoch)
    end_time = time.time()
    training_time = end_time - start_time

    # 保存模型参数数据和训练时间到列表
    model_params = {
        'model_type': config['model_type'],
        'epoch': epoch,
        'learning_rate': config['learning_rate'],
        "max_length": config['max_length'],
        "hidden_size": config['hidden_size'],
        "kernel_size": config['kernel_size'],
        "num_layers": config['num_layers'],
        "batch_size": config['batch_size'],   
        "pooling_style": config['pooling_style'],
        "optimizer": config['optimizer'],
        'accuracy': acc, 
        'training_time': training_time
        }
    model_data.append(model_params)
    
    # 保存模型参数数据和训练时间到CSV文件
    df = pd.DataFrame(model_data)
    df.to_csv(os.path.join(model_params_dir, 'model_params.csv'), index=False)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    main(Config)

    for model in ["cnn"]:
        Config["model_type"] = model
        print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    for model in ["gated_cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)