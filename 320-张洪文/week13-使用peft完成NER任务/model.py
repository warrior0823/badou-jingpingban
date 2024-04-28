import torch.nn as nn
from peft import LoraConfig, PromptEncoderConfig, PromptTuningConfig, PrefixTuningConfig, get_peft_model, TaskType
from torch.optim import SGD, Adam
from torchcrf import CRF
from transformers import AutoModelForTokenClassification


# 基于peft微调构建的bert模型
def bert_model(config):
    # 为每一个token预测一个类别。num_labels指定了分类的类别数，即NER任务中的实体数量
    model = AutoModelForTokenClassification.from_pretrained(
                                                    config["bert_path"],
                                                    return_dict=False,
                                                    num_labels=config["num_labels"],)
    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    peft_config = None
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=config["inference_mode"],
                                 r=8, lora_alpha=32, lora_dropout=0.1,
                                 target_modules=["query", "key", "value"])
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        print("模型可训练参数: ", end="")
        model.print_trainable_parameters()
    return model


class NERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = bert_model(config)
        # label的padding为-1, 不计算loss
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None):
        x_encoder = self.encoder(x)  # batch_size, seq_len, hidden_size(768)
        pred = x_encoder[0]
        if y is not None:
            return self.loss(pred.view(-1, pred.shape[-1]), y.view(-1))
        else:
            return pred

# 选择优化器
def choose_optimizer(model, config):
    if config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError('Unsupported optimizer: {}'.format(config["optimizer"]))
    return optimizer
