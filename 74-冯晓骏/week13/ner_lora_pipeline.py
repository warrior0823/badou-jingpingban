from transformers import AutoModelForTokenClassification,AutoTokenizer,pipeline
from peft import PeftModel,PeftModelForTokenClassification,get_peft_model,AutoPeftModel
import json
# 数据预处理
with open('../ner_data/schema.json', 'r', encoding='utf-8') as f:
    label_list = json.load(f)
print(label_list)
tokenizer = AutoTokenizer.from_pretrained("./hfl-chinese-macbert-base")
model = AutoModelForTokenClassification.from_pretrained("./ner_model/checkpoint-450")
model.config.id2label = {id:label for label,id in label_list.items()}
# 创建模型
model = PeftModel.from_pretrained(model,model_id="./ner_lora_model/checkpoint-135")

merge_model = model.merge_and_unload()
merge_model.save_pretrained('./ner_merge_model/')
ner_pipeline = pipeline(
    'token-classification',
    model=merge_model,
    tokenizer=tokenizer,
    aggregation_strategy='simple',
    device=0
)
x = '小明和小红在北京上班,小明和小红在北京上班,小明和小红在北京上班,小明和小红在北京上班.'
ner_result = ner_pipeline(x)
print(ner_result)
entitys = {}
for item in ner_result:
    if item['entity_group'] not in entitys :
        entitys[item['entity_group']] = []
    entitys[item['entity_group']].append(x[item['start']:item['end']])
print(entitys)