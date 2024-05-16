from config import config
import os 
from loader import load_data
from model import SiameseTripNetWork,choose_optimizer
import logging
import numpy as np

import torch

from evaluate import Evaluator

logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):

    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 定义训练数据
    train_data = load_data(config["train_data_path"],config)

    # 定义模型
    model = SiameseTripNetWork(config)

    cuda_flag =  torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()

    optimizer = choose_optimizer(config,model)

    evalutor = Evaluator(config,model,logger)

    # 
    for epoch in range(config["epoch"]):
        epoch +=1
        logger.info(f"当前开始第{epoch}轮训练")
        model.train()
        train_loss = []
        for idx,batch_data in  enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            a,p,n = batch_data
            loss = model(a,p,n,0.1)
            train_loss.append(loss.item())
            if idx % int(len(train_data)/2) ==0:
                logger.info("batch loss %f",np.mean(train_loss))
            loss.backward()
            optimizer.step()
        logger.info(f"epoch mean loss :{np.mean(train_loss)}")
        evalutor.eval(epoch)
    
    model_path = os.path.join(config["model_path"],"epoch_%d.pth" % epoch)

    torch.save(model.state_dict(),model_path)
    return 


if __name__ == "__main__":
    main(config)