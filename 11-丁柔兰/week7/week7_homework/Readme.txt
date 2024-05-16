
提问：
**  1. 数据"vocab_path":"chars.txt" 有什么作用
**  2. 将下列性能指标加入表格
    # 计算性能指标
    accuracy_lr = accuracy_score(val_data['label'], val_pred_lr)
    precision_lr = precision_score(val_data['label'], val_pred_lr)
    recall_lr = recall_score(val_data['label'], val_pred_lr)
    f1_lr = f1_score(val_data['label'], val_pred_lr)
    数据分析：正负样本数，文本平均长度等
    实验对比3种以上模型结构的分类效果
    每种模型对比模型预测速度

**  3. 需要注意使用不同模型时，应该解决的问题
    当你使用包含注意力机制（如 BERT 或其他基于 Transformer 的模型）的模型时，如果你的输入序列包含了填充（padding）的话，强烈建议你传入一个 `attention_mask`。`attention_mask` 是一个与 `input_ids` 相同大小的张量，它告诉模型哪些令牌（tokens）是重要的，哪些是填充的，应该忽略。
    下面是如何在 PyTorch 中创建 `attention_mask` 并将其传递给模型的示例：

    ```python
    import torch
    from transformers import BertTokenizer, BertModel

    # 初始化 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')

    # 示例输入文本和填充
    texts = ["你好，我叫约翰。", "嗨，你好！"]

    # 使用分词器编码文本
    # 使用 "return_tensors" 参数返回 PyTorch 张量
    # 使用 "padding=True" 自动填充到最长序列的长度
    # 使用 "truncation=True" 确保序列不会超过模型的最大长度
    encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # 从编码后的输入中提取 "input_ids" 和 "attention_mask"
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    # 将 "input_ids" 和 "attention_mask" 传给模型
    outputs = model(input_ids, attention_mask=attention_mask)

    # 输出是包含了多个返回值的元组，其中 "last_hidden_state" 是序列的隐藏状态
    last_hidden_state = outputs.last_hidden_state
    ```

    在这个例子中，我们使用了 `transformers` 库中的 `BertTokenizer` 来对中文文本进行分词，并编码为模型所需的输入格式。`attention_mask` 是在调用 `tokenizer` 的 `__call__` 方法时自动生成的，它的值会是 `1` 表示模型需要关注该位置的令牌，`0` 表示该位置是填充的，模型应该忽略。
    当你准备输入数据并送入模型进行训练或预测时，确保一起传入 `input_ids` 和对应的 `attention_mask` 是非常重要的，这样可以提高模型处理填充序列的效率和准确性。
