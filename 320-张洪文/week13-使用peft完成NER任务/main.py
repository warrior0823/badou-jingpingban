import os
import random
import numpy as np
import torch
from config import Config
from loader import load_data, load_schema
from model import NERModel, choose_optimizer
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
    # 加载数据
    train_data = load_data(config["train_path"], config)
    # 加载模型
    model = NERModel(config)

    # 标识是否使用gpu
    if config["device"] == "cuda":
        logger.info("Using GPU for training")
    model.to(config["device"])

    # 加载优化器
    optimizer = choose_optimizer(model, config)
    # 加载模型测试类
    evaluator = Evaluator(model, config, logger)
    # 训练模型
    f1 = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()  # 训练模式
        logger.info("Epoch {}/{}".format(epoch, config["epochs"]))
        watch_loss = []
        n = len(train_data) / 10
        for index, batch_data in enumerate(train_data):
            x, y = [i.to(config["device"]) for i in batch_data]
            optimizer.zero_grad()  # 梯度归0
            loss = model(x, y)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            watch_loss.append(loss.item())
            # if index % n == 0:
            # logger.info("\tstep %d/%d - loss %f" % (index, len(train_data),  loss.item()))
        macro_f1, micro_f1 = evaluator.eval(epoch)
        f1.append(macro_f1)
        f1.append(micro_f1)
        logger.info("Epoch %d/%d - loss: %.10f\n" % (epoch, config["epochs"], np.mean(watch_loss)))
    mean_f1 = np.mean(f1)
    # 是否保存模型
    if config["save_model"]:
        model_name = f'NER_{config["model"]}_{config["epochs"]}_{config["learning_rate"]}_{int(mean_f1*100)}%.pth'
        model_path = os.path.join(config["model_save_path"], model_name)
        if config["tuning_tactics"] is not None:
            saved_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
            torch.save(saved_params, model_path)
        else:
            torch.save(model.state_dict(), model_path)
    return model, train_data


if __name__ == '__main__':
    main(Config)
