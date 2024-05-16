import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

'''
修改torchdemo中的任务为多分类任务，完成训练。

'''
class TorchModel(nn.Module): 
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear=nn.Linear(input_size, 5) # 做5分类任务
        self.activation=nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()
        
    def forward(self, x, y=None):
        x=self.linear(x)
        y_pred=self.activation(x)
        
        if y is not None:
            return self.loss(y_pred, y.long())
        else:
            return y_pred
        
        
        
def build_sample():
    '''
    随机生成一个10维样本, 每个样本会被分类到 0,1,2,3,4 这5类中的一类
    '''
    
    while True:
        x=np.random.random(10)
        c=0
        i = 0
        j = len(x)-1
        while i<=j:
            if x[i]>=x[j]:
                return x, c
            else:
                i+=1
                j-=1
                c+=1
                
            
    
    


def build_dataset(total_sample_num):
    X=[]
    Y=[]
    
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)
        


def softmax(matrix:torch.Tensor):
    '''
    接收一个预测值矩阵，输出一个规范化后的矩阵
    '''
    matrix = torch.FloatTensor(matrix)
    matrix = matrix.detach().numpy()
    return np.exp(matrix)/np.sum(np.exp(matrix) ,axis=1, keepdims=True)


def to_one_hot(target, num_classes):
    print(f'type(target) = %s' % type(target))
    
    '''
    将softmax矩阵转为0-1矩阵
    '''
    
    one_hot = torch.zeros(target.shape[0], num_classes)
    one_hot.scatter_(1, target.unsqueeze(1), 1.)
    return one_hot
    
    
    



def cross_entropy(y_pred, y):
    '''
    y: target向量
    y_pred: 预测值矩阵
    '''
    batch_size, _ = y_pred.shape 
    
    y=to_one_hot(y, y_pred.shape)
    y_pred=softmax(y_pred)
    
    loss=-np.sum(y*np.log(y_pred), axis=1)
    
    return np.sum(loss)/batch_size
     
# 测试每轮模型的准确率
def evaluate(model:TorchModel):
    model.eval()
    test_sample_size=100
    test_x, test_y = build_dataset(test_sample_size)
    
    correct, wrong = 0, 0
    
    with torch.no_grad():
        test_y_pred:torch.Tensor[100, 5] = model(test_x)  # 模型预测
        for y_p, y_t in zip(test_y_pred, test_y):  # 与真实标签进行对比

            pred_label = torch.argmax(y_p).item()
            if pred_label == y_t.item():
                correct+=1
            else:
                wrong+=1
            
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
    
    
    
     

def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率
    
    model=TorchModel(input_size=input_size)
    
    optim=torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # 损失日志
    log=[]
    
    # 创建训练集
    train_x,train_y =build_dataset(train_sample)
    
    for epoch in range(epoch_num):
        model.train()
        # 本轮中，记录所有批次的损失
        watch_loss=[]
        for i in range(train_sample//batch_size):
            x=train_x[i*batch_size:(i+1)*batch_size]
            y=train_y[i*batch_size:(i+1)*batch_size]
            
            loss=model(x,y)

            loss.backward() # 计算梯度
            optim.step()  # 更新参数
            
            model.zero_grad() # 梯度归零
            
            watch_loss.append(loss.item())
            
            print(f"========\n第%d轮 平均loss = %f"%(epoch+1,np.mean(watch_loss)))
        
        acc=evaluate(model)
        log.append([acc, np.mean(watch_loss)])
    
    # 保存模型
    torch.save(model.state_dict(), "model2.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return






# 使用训练好的模型做预测
def predict(model_path, input_vecs):
    input_size=10
    model=TorchModel(input_size)
    model.load_state_dict(torch.load(f=model_path))
    print(model.state_dict)
    
    model.eval()
    with torch.no_grad():
        for input_vec in input_vecs:
            input_tensor = torch.FloatTensor([input_vec])
            result=model(input_tensor) # 获取预测结果
            pred_label=torch.argmax(result).item() # 找到预测概率最高的类别
            pred_prob=result[0][pred_label].item() # 获取该类别的预测概率
            print("输入：%s, 预测类别：%d, 概率值：%f" % (input_vec, pred_label, pred_prob))

        
        
        
if __name__ == "__main__":
    main()
    test_vec = [[10,9,8,7,6,5,2,1,3,4],
                [4,9,8,7,6,5,2,1,3,10],
                [4,3,8,7,6,5,2,1,9,10],
                [4,3,1,7,6,5,2,8,9,10]]
    predict("model2.pt", test_vec)