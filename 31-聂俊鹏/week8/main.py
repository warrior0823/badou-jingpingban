import logging
import os
import numpy as np
import torch.cuda

from config import objConfig
from loader import load_data
from model import TorchModel , choose_optimizer
from evaluate import Evaluate

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def main(mConfig):
    # 创建保存模型的目录
    if not os.path.isdir(mConfig["model_path"]):
        os.mkdir(mConfig["model_path"])
    #加载训练数据
    train_data = load_data(mConfig["train_data_path"] , mConfig)

    #加载模型
    model = TorchModel(mConfig)

    #判断是否再用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，后续操作迁移至gpu")
        model =model.cuda()

    #加载优化器
    optimizer = choose_optimizer(mConfig , model)

    #加载效果测试类
    evaluator = Evaluate(mConfig , model , logger)

    #开始训练
    for index in range(mConfig["epoch"]):
        index += 1
        model.train()
        logger.info("epoch %d begin" % index)
        train_loss = []
        for toindex ,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id , labels = batch_data
            loss = model(input_id, labels)
            train_loss.append(loss.item())
            if toindex % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(index)
    model_path = os.path.join(mConfig["model_path"], "epoch_%d.pth" % index)
    # torch.save(model.state_dict(), model_path)
    return model , train_data





if __name__ == '__main__':
    tomodel, train_datas = main(objConfig)

    def ask(question):
        input_id = train_datas.dataset.encode_sentence(question)
        tomodel.eval()
        newmodel = tomodel.cpu()
        print(torch.argmax(newmodel(torch.LongTensor([input_id]))))
