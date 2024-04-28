from collections import defaultdict
from datasets import Dataset,load_dataset
import json



def process_data_to_dataset(src_path, dst_path):

    with open('../ner_data/schema.json') as f:
        label2id = json.load(f)
    print(label2id)
    datas = defaultdict(list)
    with open(src_path, encoding='utf-8') as f:
        segments = f.read().split('\n\n')
        for index,lines in enumerate(segments):
            texts = []
            labels = []
            for line in lines.split('\n'):
                data_label = line.split(' ')
                if len(data_label) == 2:
                    texts.append(data_label[0])
                    labels.append(label2id[data_label[1]])

            datas['id'].append(index)
            datas['text'].append(texts)
            datas['ner_tag'].append(labels)

    dataset = Dataset.from_dict(datas)
    dataset.save_to_disk(dst_path)

if __name__ == '__main__':
    process_data_to_dataset('../ner_data/train','../ner_data/train_dataset')
    process_data_to_dataset('../ner_data/test','../ner_data/test_dataset')
    process_data_to_dataset('../ner_data/dev','../ner_data/dev_dataset')
    # train_dataset = Dataset.load_from_disk('../ner_data/train_dataset')
    # test_dataset = Dataset.load_from_disk('../ner_data/test_dataset')
    # print(train_dataset)
    # print(test_dataset)
