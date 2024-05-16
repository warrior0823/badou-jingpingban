import torch

from loader import load_data,generate_extend_attn_mask
from transformers import BertTokenizer
class Evaluator():
    def __init__(self,model,config):
        self.model = model
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        # self.data_loader = load_data(config,config['train_data_path'])


    def eval(self,epoch,logger,sentence):
        logger.info("Evaluating model at epoch {}".format(epoch))
        self.model.eval()
        with torch.no_grad():
            while len(sentence) < self.config['max_len']:
                input_ids = self.tokenizer.encode(sentence,truncation=True,return_tensors='pt',add_special_tokens=False)
                mask = generate_extend_attn_mask(len(input_ids[0])).int()
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    mask = mask.cuda()
                pred = self.model(input_ids,mask)
                pred_word = self.tokenizer.decode(torch.argmax(pred[0][-1],dim=-1))
                # print(pred_word)
                sentence+=pred_word
            print(sentence)

if __name__ == '__main__':
    eval = Evaluator()
    eval.eval(1,None,"让他在半年之前，就不能做出")