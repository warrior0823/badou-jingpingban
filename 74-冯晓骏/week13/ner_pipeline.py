from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

with open('../ner_data/schema.json', 'r', encoding='utf-8') as f:
    label_list = json.load(f)
print(label_list)

model = AutoModelForTokenClassification.from_pretrained("./ner_model/checkpoint-450")
model.config.id2label = {id:label for label,id in label_list.items()}
print(model.config)

tokenizer = AutoTokenizer.from_pretrained("./ner_model/checkpoint-450")


ner_pipeline = pipeline(
    'token-classification',
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy='simple',
    device=0
)
x = '小明和小红在北京上班'
ner_result = ner_pipeline(x)
print(ner_result)
entitys = {}
for item in ner_result:
    if item['entity_group'] not in entitys :
        entitys[item['entity_group']] = []
    entitys[item['entity_group']].append(x[item['start']:item['end']])
print(entitys)

'''[
    {
        'entity_group': 'LABEL_2', 
        'score': 0.97074103, 'word': 
        '小', 'start': 0, 
        'end': 1
    }, 
    {
        'entity_group': 'LABEL_6', 
        'score': 0.9146551, 
        'word': '明', 
        'start': 1, 
        'end': 2
    }, 
    {
        'entity_group': 'LABEL_8', 
        'score': 0.9991634, 
        'word': '在', 
        'start': 2, 
        'end': 3
    }, 
    {
        'entity_group': 'LABEL_0', 
        'score': 0.9946128, 
        'word': '北', 
        'start': 3, 
        'end': 4
    }, 
    {
        'entity_group': 'LABEL_4', 
        'score': 0.9961694, 
        'word': '京', 
        'start': 4, 
        'end': 5
    }, 
    {
        'entity_group': 'LABEL_8', 
        'score': 0.9988954, 
        'word': '上 班', 
        'start': 5, 
        'end': 7
    }]'''