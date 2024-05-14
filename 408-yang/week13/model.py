from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification
from config import Config
from torch.optim import Adam,SGD

TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=9, return_dict=False)

def choose_optimizer(config,model):
    optim = config["optimizer"]
    lr = config["learning_rate"]
    if optim =="sgd":
        return SGD(model.prameters(),lr =lr)
    elif optim=="adam":
        return Adam(model.parameters(),lr =lr)