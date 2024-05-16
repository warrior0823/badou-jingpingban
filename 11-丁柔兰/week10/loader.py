import pandas as pd
from sklearn.model_selection import train_test_split

# 假设你有CSV文件，其中包含两列：'review'和'label'。
data = pd.read_csv('ecommerce_reviews.csv')

# 划分训练集和验证集
train_data, valid_data = train_test_split(data, test_size=0.2)

# 数据分析
positive_samples = len(train_data[train_data['label'] == '好评'])
negative_samples = len(train_data[train_data['label'] == '差评'])
average_length = train_data['review'].apply(lambda x: len(x)).mean()

print(f"正样本数: {positive_samples}")
print(f"负样本数: {negative_samples}")
print(f"文本平均长度: {average_length}")
