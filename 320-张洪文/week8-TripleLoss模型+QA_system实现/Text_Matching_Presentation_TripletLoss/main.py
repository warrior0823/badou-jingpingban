import os
import torch
import random
import numpy as np
from config import Config
from loader import load_data, load_schema
from model import PresentationModel, choose_optimizer
from evaluate import Evaluator
import logging
# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
if Config["seed"] is not None:
    seed = Config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 模型调用入口
def main(config):
    # 创建模型保存目录
    if not os.path.isdir(config["model_save_path"]):
        os.mkdir(config["model_save_path"])
    # 每一轮训练都重新加载随机样本数据
    train_data = load_data(config["train_path"], config)
    model = PresentationModel(config)  # 加载模型
    if config["device"] == "cuda":
        logger.info("Using GPU for training")
    model.to(config["device"])
    optimizer = choose_optimizer(model, config)   # 初始化优化器
    evaluator = Evaluator(model, config, logger)  # 初始化模型测试类
    # 训练模型
    for epoch in range(1, config["epochs"]+1):
        model.train()  # 训练模式
        # logger.info("Epoch {}/{}".format(epoch, config["epochs"]))
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            a, p, n = batch_data
            a, p, n = a.to(config["device"]), p.to(config["device"]), n.to(config["device"])
            optimizer.zero_grad()  # 梯度归0
            loss = model(a, p, n)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            watch_loss.append(loss.item())
            # logger.info("\tstep %d/%d - loss %f" % (len(train_data), index+Text_Matching_Presentation, loss.item()))
        acc = evaluator.eval(epoch)
        # 忽略 nan loss
        logger.info("Epoch %d/%d - loss: %.10f\n" % (epoch, config["epochs"], np.nanmean(watch_loss)))
        if epoch != config["epochs"]:
            # 每一轮训练都重新加载随机样本数据
            train_data = load_data(config["train_path"], config)
    # 是否保存模型
    if config["save_model"]:
        model_path = os.path.join(config["model_save_path"], f'model_{config["epochs"]}.pth')
        torch.save(model.state_dict(), model_path)
    return model


if __name__ == '__main__':
    m = main(Config)
