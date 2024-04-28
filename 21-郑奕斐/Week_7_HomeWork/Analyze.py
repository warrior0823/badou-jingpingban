import pandas as pd
#Loading Data
data = pd.read_csv('文本分类练习.csv')

#Counting Good/Bad Review
grouped_data = data.groupby('label').count()
print(grouped_data)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')

#The Length Of Review Information
text_column = data['review']
text_length_describe = text_column.str.len().describe()
print(text_length_describe, 'text_length_describe')