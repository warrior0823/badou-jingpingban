from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification
import evaluate
from datasets import load_dataset, load_from_disk, DatasetDict
import json
import numpy as np
from peft import LoraConfig,TaskType,get_peft_model


def load_schema():
    with open('../ner_data/schema.json', 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema


def process_fn(examples, tokenizer=None):
    print('------------------------')
    print(len(examples))
    assert tokenizer is not None
    labels_ids_batch = []
    tokenized_examples = tokenizer(examples['text'], is_split_into_words=True, max_length=128, truncation=True)
    for i, labels in enumerate(examples['ner_tag']):
        label_ids = []
        word_ids = tokenized_examples.word_ids(batch_index=i)

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(labels[word_id])
        labels_ids_batch.append(label_ids)
    # print(len(tokenized_examples['input_ids']))
    # print(len(labels_ids_batch))
    tokenized_examples['labels'] = labels_ids_batch
    return tokenized_examples


def eval_metric(pred, schema=None, seqeval=None):
    assert seqeval is not None
    assert schema is not None
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    true_pred = [
        [list(schema.keys())[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_label = [
        [list(schema.keys())[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    result = seqeval.compute(predictions=true_pred, references=true_label, mode='strict', scheme='IOB2')
    return {
        'f1': result['overall_f1'],
        'recall': result['overall_recall'],

    }


def main():
    # 加载数据集
    train_dataset = load_from_disk('../ner_data/train_dataset')
    test_dataset = load_from_disk('../ner_data/test_dataset')
    schema = load_schema()

    dataset = DatasetDict()

    dataset['train'] = train_dataset
    dataset['test'] = test_dataset

    # 数据预处理
    tokenizer = AutoTokenizer.from_pretrained("./hfl-chinese-macbert-base")

    tokenized_dataset = dataset.map(lambda x: process_fn(x, tokenizer), batched=True)

    # 创建模型
    model = AutoModelForTokenClassification.from_pretrained("./ner_model/checkpoint-450")
    print('num_labels:', model.config.num_labels)

    #配置lora
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        target_modules=['query','key','value'],
        inference_mode=False,
        r=8,
        lora_alpha=32, #缩放为32/8=4倍
        lora_dropout=0.1,
        modules_to_save=['classifier']

    )
    model = get_peft_model(model, lora_config)
    print(lora_config)
    print(model)

    # 评估函数
    seqeval = evaluate.load("seqeval")
    print(seqeval)

    # 配置训练参数
    args = TrainingArguments(
        output_dir='./ner_lora_model',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=5,
        logging_dir='./logs',
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda pred: eval_metric(pred, schema=schema, seqeval=seqeval),
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
