# coding=utf-8
import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import json


'''
    nlp 多分类任务 ，判断a 出现在第几个位置
    现有模型或者是rnn模型来实现 
'''


class TorchModel(nn.Module):
    def __init__(self, embedding_dim,seq_len,vocab) -> None:
        super().__init__()
        # 按照词表大小生成向量
        self.embedding = nn.Embedding(num_embeddings=len(vocab),embedding_dim=embedding_dim)
        # self.pool = nn.AvgPool1d(seq_len) # 在seq_len方向进行均值池化
        self.layer = nn.RNN(input_size=embedding_dim,hidden_size=2*embedding_dim,batch_first=True,bias=False)
        self.classify = nn.Linear(2*embedding_dim,seq_len+1) #输出结果为seq_len+1维
        self.loss = nn.functional.cross_entropy # 均方差loss

    def forward(self,x,y=None):
        x = self.embedding(x) #(batch_size,seq_len)->(batch_size,seq_len,embeding_dim)
        # rnn 不需要再做pooling了，把元素加进去即可
        # x = x.transpose(1,2)  #(batch_size,seq_len,embed_dim)->(batch_size,embedding_dim,seq_len)
        # 对长度方向进行池化，所以需要先进行转置操作
        # x = self.pool(x) # (batch_size,embedding_dim,1)->(batch_size,embedding_dim,1)
        # x = x.squeeze() # 去掉最后的一维的维度 #(batch_size,embedding_dim,1) -> (batch_size,embedding_dim)
        y_pred,_ = self.layer(x) #(batch_size,embedding_dim)->(batch_size,2*embedding_dim)
        # 取rnn 的最后一个的结果
        y_pred = y_pred[:,-1,:]
        y_pred = self.classify(y_pred) # (batch_size,2*embedding_dim)->(batch_size,1)
        if y is not None:
            # print('y shape',y.shape)
            return self.loss(y_pred,y)
        else :
            return y_pred  #返回预测值

def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab ={"pad":0}
    for index,char in enumerate(chars):
        vocab[char] = index+1
    vocab["unk"] = len(vocab)
    return vocab

# 随机生成训练数据，如果字符串中有a ,返回1，否则返回0
def build_sample(vocab,seq_len):
    x = [random.choice(list(vocab.keys())) for _ in range(seq_len)]
    # y = x.index('a')
    if 'a' in x:
        y = x.index('a')
    else:
        y= seq_len
    x = [vocab.get(word,vocab["unk"]) for word in x] #将字符转为embed 的索引
    return x,y

def build_dataset(sample_num,vocab,seq_len):
    dataset_x = []
    dataset_y = []
    y_cnt = len(dataset_y)
    a_cnt = 0
    while y_cnt<=sample_num:
    # for i in range(sample_num):
        x,y = build_sample(vocab,seq_len=seq_len)
        if a_cnt >2 and y == seq_len:
            continue
        dataset_x.append(x)
        dataset_y.append(y)
        y_cnt += 1
        if y==seq_len:
            a_cnt += 1
       
    count = dataset_y.count(seq_len)
    print('不含a的样本个数为：%d' %count)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

def build_model(vocab,embedding_dim,seq_len):
    model = TorchModel(embedding_dim,seq_len,vocab=vocab)
    return model

def evaluate(model,vocab,seq_length):
    model.eval()
    test_sample =200
    eva_x,eva_y = build_dataset(test_sample,vocab,seq_len=seq_length)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(eva_x) # 模型预测，批量方式
        for y_p,y_t in zip(y_pred,eva_y):
            print("y_p,y_t", torch.argmax(y_p),int(y_t))
            if torch.argmax(y_p) == int(y_t) :
                correct += 1
            else:
                wrong +=1
    print("样本总数为:%d,正确样本个数为%d,正确率为: %f" % (correct+wrong,correct,correct/(correct+wrong)))
    return correct/(correct+wrong)




def train():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    embedding_dim = 20
    seq_length = 6
    lr = 0.005

    vocab = build_vocab()
    
    model = build_model(vocab,embedding_dim,seq_len=seq_length)

    optim = torch.optim.Adam(model.parameters(),lr = lr)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            train_x,train_y  =build_dataset(batch_size,vocab,seq_length)
            optim.zero_grad()
            model_loss = model(train_x,train_y) #计算loss
            model_loss.backward() #计算梯度
            optim.step() # 更新权重
            watch_loss.append(model_loss.item())
        print("batch %d 平均loss 为:%f \n" %(epoch+1,np.mean(watch_loss)))
        # 训练完一轮之后进行evaluate
        acc = evaluate(model,vocab,seq_length)
        log.append([acc,np.mean(watch_loss)])
    
    #画图
    plt.plot(range(len(log)),[l[0]for l in log],label="acc",color="red")
    plt.plot(range(len(log)),[l[1]for l in log],label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(),'./model/nlp_multi_class_rnn.pth')

    # 保存词表
    writer = open('./data/vocab.json',"w",encoding="utf8")
    writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
    writer.close()
    return 

def predict(model_path,vocab_path,input_strings):
    # 加载词表，评价预测准确率
    embedding_dim = 20 # 每个字的维度，可以不和训练的时候一致吗？ 【应该是不行的】
    seq_length = 6 #测试是否可以和训练的时候不一致

    vocab = json.load(open(vocab_path,"r",encoding="utf8"))
    model = build_model(vocab,embedding_dim,seq_len=seq_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char,vocab["unk"]) for char in input_string])
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.LongTensor(x))
    for vec,res in zip(input_strings,y_pred):
        predicted = torch.argmax(res)
        print("输入%s,预测类别：%d" %(vec,predicted))
    

if __name__ =="__main__":
   
    # y = 'nakwww'.index('a')
    # print(y)
    train()
    test_strings =["fnvfee", "wzadfg", "rqwdeg", "nakwww"]
    predict('./model/nlp_multi_class_rnn.pth','./data/vocab.json',test_strings)

