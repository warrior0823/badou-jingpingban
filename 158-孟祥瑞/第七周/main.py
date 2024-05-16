# coding: utf-8

import torch
import numpy as np
import random
import os
from config import Config
import logging
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" %epoch)
    # torch.save(model.state_dict(), model_path)
    return acc


if __name__ == '__main__':
    main(Config)
