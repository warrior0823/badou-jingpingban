# coding=utf8
'''
    计算bert的参数量
'''
from transformers import BertModel

def cal_params_cnt(state_dict,num_layers):
        embedding_cnt = get_embedding_parm_cnt(state_dict)
        attention_layer_cnt = 0
        for i in range(num_layers):
            attention_layer_cnt += get_attention_layer_param_cnt(state_dict,i)
        pool_param_cnt = get_pool_param_cnt(state_dict)
        return embedding_cnt+attention_layer_cnt+pool_param_cnt
        
def get_pool_param_cnt(state_dict):
    pool_param_cnt =0 
    m,n =  state_dict["pooler.dense.weight"].shape
    pool_param_cnt+= m*n
    m = state_dict["pooler.dense.bias"].shape
    pool_param_cnt+= int(m[0])
    return pool_param_cnt

def get_attention_layer_param_cnt(state_dict,i):
    attent_layer_param_cnt = 0
    m,n=  state_dict["encoder.layer.%d.attention.self.query.weight"%i].shape
    attent_layer_param_cnt += m*n
    m= state_dict["encoder.layer.%d.attention.self.query.bias" % i].shape
    attent_layer_param_cnt += int(m[0])
    m,n= state_dict["encoder.layer.%d.attention.self.key.weight" % i].shape
    attent_layer_param_cnt += m*n
    m= state_dict["encoder.layer.%d.attention.self.key.bias" % i].shape
    attent_layer_param_cnt += int(m[0])
    m,n=state_dict["encoder.layer.%d.attention.self.value.weight" % i].shape
    attent_layer_param_cnt += m*n
    m=  state_dict["encoder.layer.%d.attention.self.value.bias" % i].shape
    attent_layer_param_cnt += int(m[0])
    # attention ouput 层
    m,n= state_dict["encoder.layer.%d.attention.output.dense.weight"%i].shape
    attent_layer_param_cnt += m*n
    m= state_dict["encoder.layer.%d.attention.output.dense.bias"%i].shape
    attent_layer_param_cnt += int(m[0])

    # attention之后经过一次layerNorm (add&norm) 残差网络
    m= state_dict["encoder.layer.%d.attention.output.LayerNorm.weight"%i].shape
    attent_layer_param_cnt += int(m[0])
    m = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias"%i].shape
    attent_layer_param_cnt += int(m[0])

    # feedforward 4倍数 两个线性层
    m,n= state_dict["encoder.layer.%d.intermediate.dense.weight"%i].shape 
    attent_layer_param_cnt += m*n

    m=  state_dict["encoder.layer.%d.intermediate.dense.bias"%i].shape
    attent_layer_param_cnt += int(m[0])

    m,n=  state_dict["encoder.layer.%d.output.dense.weight"%i].shape
    attent_layer_param_cnt += m*n
    m= state_dict["encoder.layer.%d.output.dense.bias" %i].shape
    attent_layer_param_cnt += int(m[0])
    
    # 再经过add&norm 残差网络
    m= state_dict["encoder.layer.%d.output.LayerNorm.weight"%i].shape
    attent_layer_param_cnt += int(m[0])
    m= state_dict["encoder.layer.%d.output.LayerNorm.bias"%i].shape
    attent_layer_param_cnt += int(m[0])
    return attent_layer_param_cnt


def get_embedding_parm_cnt(state_dict):
    embedding_cnt = 0

    word_embedding_m ,word_embedding_n = state_dict["embeddings.word_embeddings.weight"].shape
    embedding_cnt += word_embedding_m*word_embedding_n
    
    position_embedding_m, position_embedding_n = state_dict["embeddings.position_embeddings.weight"].shape
    embedding_cnt += position_embedding_m*position_embedding_n
    
    token_type_embedding_m, token_type_embedding_n =  state_dict["embeddings.token_type_embeddings.weight"].shape
    embedding_cnt += token_type_embedding_m * token_type_embedding_n
    
    embeddings_layer_norm_weight_m = state_dict["embeddings.LayerNorm.weight"].shape
    embedding_cnt += int(embeddings_layer_norm_weight_m[0])

    embeddings_layer_norm_bias_m  = state_dict["embeddings.LayerNorm.bias"].shape
    embedding_cnt += int(embeddings_layer_norm_bias_m[0])
    return embedding_cnt

def cal_params_cnt_no_dict(vocab,max_sequence_length,embedding_size,token_type=2,num_layer=12):
    # embedding 层
    
    embedding_cnt = vocab*embedding_size + max_sequence_length*embedding_size+ token_type*embedding_size + embedding_size+embedding_size
    # attention层
    self_attention_cnt = (embedding_size*embedding_size + embedding_size)*3 

    attention_output_cnt = embedding_size*embedding_size + embedding_size
    # wx+b
    attention_layer_norm_cnt = embedding_size + embedding_size
    # np.dot()
    ffn_cnt = embedding_size * embedding_size*4 +embedding_size*4 + 4*embedding_size*embedding_size+embedding_size
    # wx+b
    layer_norm_cnt_2 = embedding_size + embedding_size

    # pool_cnt np.dot()
    pool_cnt = embedding_size*embedding_size + embedding_size

    total_cnt = embedding_cnt + num_layer*(self_attention_cnt+attention_output_cnt+attention_layer_norm_cnt+ffn_cnt+layer_norm_cnt_2) + pool_cnt

    return total_cnt


def main():
    vocab = 21128               # 词表数目
    max_sequence_length = 512   # 最大句子长度
    embedding_size = 768
    bert =BertModel.from_pretrained(f"./bert/model/bert_base_chinese", return_dict=False)
    state_dict = bert.state_dict()
    # print(state_dict.keys())
    parameter_count = 0
    for name, param in state_dict.items():
    # 忽略掉不包含'numel'方法的键，这通常是指向其他模块的参数
        if hasattr(param, 'numel'):
        # 累加该参数的数量
            parameter_count += param.numel()
    print(parameter_count)
    
    # print(bert.state_dict().keys()) 
    # diyBert = DiyBert(state_dict=state_dict)
    param_cnt = cal_params_cnt(state_dict,num_layers=12)
    print(param_cnt)
    param_cnt_new = cal_params_cnt_no_dict(vocab,max_sequence_length,embedding_size,token_type=2,num_layer=12)
    print(param_cnt_new)

    
if __name__ == "__main__":
   main()


