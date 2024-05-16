# -*- coding: utf-8 -*-
import pandas as pd
import torch
import random
import os
import numpy as np
import logging
from config_hw import Config
from model_hw import TorchModel, choose_optimizer
from evaluate_hw import Evaluator
from loader_hw import load_data
from collections import defaultdict
from datetime import datetime
import openpyxl

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

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["data_path"], config)
    # print(train_data)
    #加载模型
    model = TorchModel(config)
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
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
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)
    # fast_text,lstm,bert,bert_lstm
    # for model in ["cnn",""]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    d_ls=defaultdict(list)
    # "bert" 单独训练
    # for model in ["fast_text","cnn","lstm"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128, 256]:
    #             Config["hidden_size"] = hidden_size
    #             t1=datetime.now()
    #             acc=main(Config)
    #             t2=datetime.now()
    #             print("最后一轮准确率：", acc, "当前配置：", Config)
    #             d_ls["model_type"].append(Config["model_type"])
    #             d_ls["max_length"].append(Config["max_length"])
    #             d_ls["hidden_size"].append(Config["hidden_size"])
    #             d_ls["batch_size"].append(Config["batch_size"])
    #             d_ls["pooling_style"].append(Config["pooling_style"])
    #             d_ls["learning_rate"].append(Config["learning_rate"])
    #             d_ls["time"].append(str(t2-t1))
    #             d_ls["acc"].append(acc)
    # df=pd.DataFrame(d_ls)
    # df.to_excel("outpu.xlsx",index=False,engine="openpyxl")
    Config["model_type"] = "bert"
    d_ls["learning_rate"]= 1e-4
    t1 = datetime.now()
    acc = main(Config)
    t2 = datetime.now()
    print("最后一轮准确率：", acc, "当前配置：", Config)
    d_ls["model_type"].append(Config["model_type"])
    d_ls["max_length"].append(Config["max_length"])
    d_ls["hidden_size"].append(Config["hidden_size"])
    d_ls["batch_size"].append(Config["batch_size"])
    d_ls["pooling_style"].append(Config["pooling_style"])
    d_ls["learning_rate"].append(Config["learning_rate"])
    d_ls["time"].append(str(t2 - t1))
    d_ls["acc"].append(acc)
    df = pd.DataFrame(d_ls)
    df.to_excel("outpu.xlsx", index=False, engine="openpyxl")




