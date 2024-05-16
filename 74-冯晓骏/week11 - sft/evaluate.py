import heapq
import random

import torch

from loader import load_data,generate_extend_attn_mask_for_2_seq
from transformers import BertTokenizer, BertModel


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
            while len(sentence) <= 510:
                src_result = self.tokenizer.encode(sentence, return_tensors='pt',
                                                        add_special_tokens=True,
                                                        )

                # print(src_result)
                # print(self.tokenizer.decode(src_result[0]))

                src_result = src_result.cuda()
                pred = self.model(src_result)
                pred_word = self.tokenizer.decode(torch.argmax(pred[0][-1]))
                sentence += pred_word
                if pred_word == '[SEP]':
                    # print('pred word is sep!!')
                    break

            print(sentence)



        # input()
            # print(self.tokenizer.decode(src_result['input_ids'][0]))
            # print(src_result['attention_mask'].shape)
            # extend_attn_mask = generate_extend_attn_mask_for_2_seq(len(sentence) + 2,
            #                                                        self.config['max_len'] - len(sentence) - 2)
            # print(extend_attn_mask.shape)
            # extend_attn_mask = extend_attn_mask.bool() & src_result['attention_mask'].bool()
            # extend_attn_mask = extend_attn_mask.int()

            # input_id = src_result['input_ids']



if __name__ == '__main__':
    from config import Config
    eval = Evaluator(None,Config)
    eval.eval(1,None,"今天下午，中新网创业精英荟来到北京石榴中心举办了活动，众多潮流公司展示了潮人健身日常：平衡车、攀岩、智能单车…小编为大家采访到攻防箭的工作人员，给大家展示时下最潮的健身方式——射箭！你会考虑这种方式健身吗？秒拍视频")
