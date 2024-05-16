import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练，实现一个自行构造的找规律（机器学习）任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标

"""
class MutiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MutiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) #线性层
        self.loss=nn.functional.cross_entropy  #loss函数采用交叉熵损失
    #当输入真实标签时，返回loss值，无真实标签时，返回预测值
    def forward(self, x,y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.long())  #预测值和真实值loss
        else:
            return y_pred  #输出预测结果

#生成一个样本，样本的生成方法，代表了我们要学习的规律
#随机生成一个5维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample():
    x=np.random.random(5)
    #获取最大值的索引
    max_index=np.argmax(x)
    if max_index==0:
        return x,0
    elif max_index==1:
        return x,1
    elif max_index==2:
        return x,2
    elif max_index==3:
        return x,3
    else:
        return x,4

#随机生成一批样本，正负样本均匀生成
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)
#测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(x) #模型预测
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p)==int(y_t):  #与真实标签进行对比
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d,正确率：%f" %(correct,correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    #配置参数
    input_size = 5  #输入向量维度
    epoch_num=20   #训练轮数
    batch_size=20  #每次训练样本个数
    train_sample_num=5000  ##训练样本总数
    learning_rate=0.01  #学习率
    #建立模型
    model = MutiClassficationModel(input_size)
    #选择优化器
    optim=torch.optim.Adam(model.parameters(), lr=learning_rate)
    log=[]
    #创建训练集，正常任务是读取训练集
    train_x,train_y=build_dataset(train_sample_num)
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch_index in range(train_sample_num//batch_size):
            x=train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss=model(x,y) #计算loss
            loss.backward() #计算梯度
            optim.step()    #更新权重
            optim.zero_grad() #梯度归零
            watch_loss.append(loss.item())
        print("========\n第%d轮平均loss:%f" % (epoch+1,np.mean(watch_loss)))
        acc=evaluate(model) #测试本轮模型结果
        log.append([acc,float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(),"model.pt")
    #画图
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label="acc")  #画acc曲线
    plt.plot(range(len(log)),[l[1] for l in log],label="loss") #画loss曲线
    plt.legend()
    plt.show()
    return

#使用训练好的模型做预测
def predict(model_path,input_vec):
    input_size=5
    model=MutiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path)) #加载训练好的权重
    print(model.state_dict())

    model.eval()  #测试模式
    with torch.no_grad(): #不计算梯度
        result=model.forward(torch.FloatTensor(input_vec))  #模型预测
    for vec,res in zip (input_vec,result):
        print("输入：%s,预测类别:%s,概率值：%s" %(vec,torch.argmax(res),res)) #打印结果

if __name__ == '__main__':
    main()
    test_vec =[[0.24414119, 0.56233458, 0.2823562 , 0.81050553, 0.14151241],
                [0.520003  , 0.38263849, 0.69325201, 0.33531836, 0.50717767],
                [0.59026669, 0.70001196, 0.22383931, 0.72356643, 0.49559008],
                [0.76805601, 0.66605479, 0.0201124 , 0.19872958, 0.10380272]]
    predict("model.pt",test_vec)
#随机生成一个4*5的向量，浮点数
