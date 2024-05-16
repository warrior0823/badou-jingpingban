import torch
import torch.nn as nn
import numpy as np

'''
这段代码首先使用 PyTorch 创建了一个一维卷积层（nn.Conv1d），
然后定义了一个函数 numpy_cnn1d 来使用 NumPy 手动实现一维卷积的前向计算过程。
最终，通过比较 PyTorch 卷积层的输出与 NumPy 手动计算的输出来验证两者是否一致，这有助于理解卷积操作的内部运作原理。
'''

'''
代码创建了一个 PyTorch 的 1 维卷积层 torch_cnn1d，该层将输入的通道数从 input_dim 转换为 hidden_size，并使用大小为 kernel_size 的卷积核。
为了展示卷积计算的过程，代码定义了一个 numpy_cnn1d 函数，使用 NumPy 实现了相同的卷积操作。
代码使用 unsqueeze(0) 方法将输入张量 x 添加一个批次维度，从而匹配 PyTorch 卷积层的输入要求。
最后，代码通过比较 PyTorch 和 NumPy 实现的输出来验证卷积计算的正确性。
'''

'''
注意：

在 PyTorch 中，nn.Conv1d 的权重形状为 (output_channels, input_channels, kernel_size)，而在 NumPy 实现中，权重形状是 (output_channels, input_channels * kernel_size)。
输入张量 x 的形状在 PyTorch 卷积层中应该是 (batch_size, input_channels, length)，但在这个代码示例中，由于 x 只有两个维度，因此需要使用 unsqueeze(0) 方法添加一个批次维度。
NumPy 实现的卷积计算中，窗口移动的步长固定为 1，不支持其他步长或填充。
'''

# 使用 PyTorch 创建一个一维卷积层
input_dim = 6
hidden_size = 8
kernel_size = 2
torch_cnn1d = nn.Conv1d(input_dim, hidden_size, kernel_size)
# 输出卷积层的权重和偏置的形状
for key, weight in torch_cnn1d.state_dict().items():
    print(key, weight.shape)

# 随机初始化一个输入张量 x
x = torch.rand((6, 8))  # embedding_size * max_length

# 定义一个函数，使用 NumPy 手动实现一维卷积的计算过程
def numpy_cnn1d(x, state_dict):
    weight = state_dict["weight"].numpy()  # 提取卷积层的权重
    bias = state_dict["bias"].numpy()  # 提取卷积层的偏置
    sequence_output = []  # 存储卷积操作的输出
    # 遍历输入张量的每个窗口
    for i in range(0, x.shape[1] - kernel_size + 1):
        window = x[:, i:i+kernel_size]  # 提取当前窗口的数据
        kernel_outputs = []  # 存储当前窗口的卷积结果
        # 对于卷积核中的每个过滤器
        for kernel in weight:
            # 计算过滤器与当前窗口的点积，并将结果存储起来
            kernel_outputs.append(np.sum(kernel * window))
        # 将当前窗口的卷积结果加上偏置，并追加到输出列表中
        sequence_output.append(np.array(kernel_outputs) + bias)
    # 将输出列表转换为 NumPy 数组，并转置以匹配 PyTorch 卷积层的输出形状
    return np.array(sequence_output).T

# 打印输入张量的形状
print(x.shape)
# 使用 PyTorch 卷积层对输入张量进行卷积，并打印结果
print(torch_cnn1d(x.unsqueeze(0)))
# 打印 PyTorch 卷积层输出的形状
print(torch_cnn1d(x.unsqueeze(0)).shape)
# 使用 NumPy 手动实现的一维卷积对输入张量进行卷积，并打印结果
print(numpy_cnn1d(x.numpy(), torch_cnn1d.state_dict()))
