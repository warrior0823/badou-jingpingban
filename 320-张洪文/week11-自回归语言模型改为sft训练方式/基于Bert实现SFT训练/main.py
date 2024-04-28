import os
import random
import numpy as np
import torch
from config import Config
from loader import load_data
from model import LanguageModel, choose_optimizer
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
    model = LanguageModel(config)
    if config["device"] == "cuda":
        logger.info("Using GPU for training")
    model.to(config["device"])
    optimizer = choose_optimizer(model, config)  # 加载优化器
    evaluator = Evaluator(model, config, train_data.dataset.tokenizer)  # 加载模型测试类

    # 训练模型
    for epoch in range(1, config["epochs"]+1):
        model.train()  # 训练模式
        # logger.info("Epoch {}/{}".format(epoch, config["epochs"]))
        watch_loss = []
        # print_loss_number = int(len(train_data) // 10)
        for index, batch_data in enumerate(train_data):
            x, mask, y = [i.to(config["device"]) for i in batch_data]
            optimizer.zero_grad()  # 梯度归0
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            watch_loss.append(loss.item())
            # if index % print_loss_number == 0:
            # logger.info("\tstep %d/%d - loss %f" % (index, len(train_data),  loss.item()))

        print(evaluator.eval("北京明年拟推工作日半价观看电影"))
        # print(evaluator.eval("南京一合金厂锅炉发生爆炸"))
        logger.info("Epoch %d/%d - loss: %f\n" % (epoch, config["epochs"], np.mean(watch_loss)))
    print("18日北京广电局电影处处长王健表示，北京明年有望从周一到周五每天上午或下午推出统一半价观影时段，还将要求影票上电影时间须为剔除片前广告的时间。首都影院联盟若7成以上会员同意，将实施此优惠政策（京华时报） ")
    # 是否保存模型
    if config["save_model"]:
        model_path = os.path.join(config["model_save_path"], f'NER_{config["model"]}_{config["epochs"]}.pth')
        torch.save(model.state_dict(), model_path)
    return model


if __name__ == '__main__':
    main(Config)

