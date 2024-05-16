import math
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from config import Config
from loader import data_load
from model import TorchModel, choose_optimizer
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
    train_data, test_data, pred_data = data_load(config)
    # 加载模型
    model = TorchModel(config)
    # if config["device"] == "cuda":
    #     logger.info("Using GPU for training")
    model.to(config["device"])
    # 加载优化器
    optimizer = choose_optimizer(model, config)
    # 加载模型测试类
    evaluator = Evaluator(model, config, logger)
    # 训练模型
    train_x, train_y = train_data
    train_x, train_y = train_x.to(config["device"]), train_y.to(config["device"])
    acc_end = 0.0
    for epoch in range(1, config["epochs"]+1):
        model.train()  # 训练模式
        # logger.info("Model: {}  Epoch {}/{}".format(config["model"], epoch, config["epochs"]))
        watch_loss = []
        batch_num = math.ceil(len(train_x)/config["batch_size"])
        for i in range(batch_num):
            x = train_x[i*config["batch_size"]: (i+1)*config["batch_size"]]
            y = train_y[i * config["batch_size"]: (i + 1) * config["batch_size"]]
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            # if i % 10 == 0:
            #     logger.info("\tstep %d/%d - loss %f" % (i, batch_num, loss.item()))
        acc = evaluator.evaluate(test_data)
        acc_end = max(acc_end, acc)
        # logger.info("\tEpoch %d/%d - acc: %.4f - loss: %f" % (epoch, config["epochs"], acc, np.mean(watch_loss)))

    start_time = time.time()
    pred_acc = evaluator.evaluate(pred_data)
    pred_time = time.time() - start_time
    return model, acc_end, pred_time, pred_acc


def test_model(save_model=True):
    column_name = ["Model", "Epochs", "Batch Size", "Hidden Size", "Learning Rate",
                   "Accuracy", "Predict 1000 Acc", "Predict 1000 Time"]
    # results_dict = defaultdict(list)
    results = []
    models = []
    # 超参数的网格搜索用来对比所有模型: 中间日志可以关掉，避免输出过多信息.
    model_list = ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "rcnn", "bert", "bert_lstm",
                  "bert_cnn", "bert_mid_layer"]
    model_list2 = ["lstm", "gru", "rnn", "cnn", "gated_cnn", "rcnn", "bert", "bert_cnn"]
    # model_list2 = ["fast_text"]
    for model in model_list2:
        Config["model"] = model
        for epoch in [10, 20]:
            Config["epochs"] = epoch
            for batch_size in [256, 512]:
                Config["batch_size"] = batch_size
                for hidden_size in [128, 256]:
                    Config["hidden_size"] = hidden_size
                    for lr in [1e-3, 1e-4]:
                        Config["learning_rate"] = lr
                        train_model, acc_end, pred_time, pred_acc = main(Config)
                        print("准确率: %.4f, 预测1000条的时间: %.4f, 预测1000条的准确率: %.4f  " % (acc_end, pred_time, pred_acc), end="")
                        print("模型: %s, 轮次: %d, 批次大小: %d, 隐藏层大小: %d, 学习率: %f" % (model, epoch, batch_size, hidden_size, lr))
                        results.append([model, str(epoch), str(batch_size), str(hidden_size), str(lr), str(acc_end), str(pred_acc), str(pred_time)])
                        models.append(train_model)
    with open("results.csv", "w", encoding="utf8") as f:
        f.write(",".join(column_name) + "\n")  # 写入标题
        for i in results:
            f.write(",".join(i) + "\n")  # 写入内容
    # 保存模型
    if save_model:
        print("保存模型...")
        model_path = Config["model_save_path"]
        for i in range(len(model_list2)):
            result = results[i*16: (i+1)*16]
            max_index = np.argmax([i[5] for i in result])
            torch.save(models[max_index], f"{model_path}{result[max_index][0]}_{result[max_index][1]}_{result[max_index][2]}_{result[max_index][3]}_{result[max_index][4]}.pth")


if __name__ == '__main__':
    # main(Config)
    test_model()
