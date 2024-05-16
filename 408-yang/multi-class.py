import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
"""
整体训练流程，预测一个多分类任务
"""

# 定义模型结构
class TorchModule(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=3),
            nn.ReLU()
        )
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        y_pred = self.layers(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
        
# 定义数据集.单个数据集
def build_sample():
    x = np.random.random(3)
    # if(x[0]<=x[1] and x[0] <= x[2]):
    #     return x, 0
    # elif(x[0]>x[1] and x[0] <=x[2]):
    #     return x, 1
    # elif (x[0]>x[1] and x[0] >x[2]):
    #     return x,2
    if np.argmax(x)==0:
        return x,0
    elif np.argmax(x)==1:
        return x,1
    else :
        return x,2

def build_dataset(total_sample_num):
    X = []
    Y = []
    count_0 = 0
    count_1 = 0 
    count_2 = 0
    while len(Y) < total_sample_num:
    # for i in range(total_sample_num):
        x,y = build_sample()
        if  y==0 and count_0>(total_sample_num/3 +2):
            continue
        elif y==1 and count_1>(total_sample_num/3 +2):
            continue
        elif y==2 and count_2>(total_sample_num/3 +1):
            continue
        X.append(x)
        Y.append(y)
        if y==0:
            count_0 += 1
        if y==1:
            count_1 += 1
        if y==2:
            count_2 += 1
    element_counts=Counter(Y)
    most_common_elements = element_counts.most_common(3)
    print(most_common_elements)
    return torch.FloatTensor(X),torch.LongTensor(Y)


# 定义评估方法，用测试机|验证集的数据来评估模型当前已经训练出来的权重对数据的评估效果
def evaluate(model):
    # 正确的数的个数
    model.eval()
    test_sample_num = 100
    test_x,test_y = build_dataset(test_sample_num)
    # 使用unique方法找出所有唯一元素及其索引
    # _, unique_indices = test_y.unique(return_inverse=True)

    # 计算每个元素的出现次数
    # counts = torch.bincount(unique_indices)
    # 打印出每个元素及其出现次数
    # for idx, count in enumerate(counts):
    #     print(f"Element {test_y[unique_indices == idx][0]}: {count}")
    correct,total = 0,0
    with torch.no_grad(): # 需要在不计算梯度的情况下进行评估
        # model中已经有了各个权重的值了，x带入进行进行得到预测的y值，然后进行评估
        y_pred = model(test_x)
        for y_p ,y_t in zip(y_pred,test_y):
            # 返回值有两个，第一个是最大的值，第二个是最大值的索引
            # print("y_p: ",y_p)
            _,predicted = torch.max(y_pred.data,dim=1)
            # print("_: ",_)
            # print("test_y: ", test_y)
            # print("predicted: ",predicted)
            total = test_y.size(0)
            correct = (predicted == test_y).sum().item()
            item =0
            correct_0 = len([item for item1, item2 in zip(predicted, test_y) if item1 == 0 and item2 == 0])
            correct_1 = len([item for item1, item2 in zip(predicted, test_y) if item1 ==1 and item2 == 1])
            correct_2 = len([item for item1, item2 in zip(predicted, test_y) if item1 ==2 and item2 == 2])
    print("正确预测个数为:%d,correct_0:%d,correct_1:%d,correct_2:%d,total:%d,正确率:%f" %(correct,correct_0,correct_1,correct_2,total,correct/total ))
    return correct/(total) 
        


# 定义训练方法
def  train():
    epoch_num =20
    batch_size = 50
    total_sample_num = 5000
    input_size = 3
    lr = 1e-2
    model = TorchModule(input_size)
    optim = torch.optim.Adam(model.parameters(),lr = lr)
    log = []
    train_x,train_y  = build_dataset(total_sample_num)
    watch_loss = []
    for epoch in range(epoch_num):
        model.train()
        for batch_index in range(total_sample_num//batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1) * batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1) * batch_size]
            loss = model(x,y)
            loss.backward() # 反向传播计算梯度，相当于是求导的步骤
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归
            # mse batch批量计算的时候，本身已经除以batch_size 了
            watch_loss.append(loss.item())
        print(f"当前训练到第{epoch}轮,平均loss为:",np.mean(watch_loss))
        acc = evaluate(model) # 测试本轮模型效果
        log.append([acc,float(np.mean(watch_loss))])
    torch.save(model.state_dict(),'./model/model-multi-class.pt')
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log],label="acc",color="red")
    plt.plot(range(len(log)), [l[1] for l in log],label="loss")
    plt.show()
    return 

def predict(model_path,input_vec):
    input_size = 3
    model = TorchModule(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    _, predicted_all = torch.max(result, 1)
    print(predicted_all)
    for vec,res in zip(input_vec,result):
        print('1111')
        print(res)
        print('2222')
        _,predicted_1 = torch.max(result.data,dim=1)
        print('333')
        predicted = torch.argmax(res)
        # 将预测结果添加到列表中
        # predictions.extend(predicted.tolist())
        print("输入%s,预测类别：%d" %(vec,predicted))
        print(predicted_1)
  

    


if __name__ =="__main__":
    # train()
    test_vec = [[0.07889086,0.15229675,0.31082123],
                [0.94963533,0.5524256,0.95758807],
                [0.78797868,0.67482528,0.13625847],
                [0.59416669,0.19349776,0.92579291]]
    predict("./model/model-multi-class.pt",test_vec)
    # build_dataset(5000)