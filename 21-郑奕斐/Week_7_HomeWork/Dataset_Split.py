import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('文本分类练习.csv')

X = data['review']
y = data['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
train_data = pd.DataFrame({'review': X_train, 'label': y_train})
val_data = pd.DataFrame({'review': X_val, 'label': y_val})

train_data.to_json('train_data.json', orient='records', lines=True,force_ascii=False)
val_data.to_json('val_data.json', orient='records', lines=True, force_ascii=False)