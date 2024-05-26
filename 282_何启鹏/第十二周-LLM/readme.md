## ChatGLM

```
chatglm首先要说起它的基座GLM,GLM既可以做Encoder也可以做Decoder,主要通过两种mask方式来实现：
[mask]:bert形式，随机mask文本中的短span
[gmask]:gpt形式，mask未尾的长span
在chatglm里面做生成任务时，是用[gmask]。chaglm2中完全采用gmask来进行预训练
在ChatGLM的内部结构中的变换，从下到上依次是
位置编码：从BERT的训练式位置编码转变为旋转位置编码
激活函数：从BERT中的GeLU转变为GLU,在ChatGLM2中又变成了SwiGLU

LayerNormalization:采用的是DeepNorm,是对post-Normalization的改进，即在残差之后做Normalization。在ChatGLM中，把layer-normalization改为RMS-Normalization.

在ChatGLM2.0中还添加了一些其他变化：
1.FlashAttenion:利用显存和内存来做加速
2.Multi--Query Attention:多个头只采用一个KV对，通过参数共享来降低显存占用
```



## baichuan

```
从下到上依次是：
位置编码：7B采用的是旋转位置编码，13B采用的是ALIB引位置编码
激活函数：SwiGLU
Normalization:RMS Norm,对每个layer的输入进行normalization,输入和attention残差之后进行normalization,
残差：输入和attention残差，attention和mlp输出残差
其他优化：memory efficient attention
```



## LLaMA

```
首先从数据角度，lamma2.0为2.0T,lamma1是1.4T.
其次是上下文长度，lamma1是2k,lamma2是4k.
最后从模型架构角度：从整体上看，二者都遵循自回归Transformer的架构，只不过内部的细节有些不同。
位置编码：二者都采用了旋转位置编码
Normalization:二者都采用pre-normalization,只不过1.0中采用原始的LayerNormalization,2.0中采用的是RMSNorm
激活函数：采用SwiGLU
```

