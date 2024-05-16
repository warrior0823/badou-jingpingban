import torch
#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载&清洗语料
def load_and_clean_corpus(corpus_path):
    cleaned_corpus = ""
    with open(corpus_path, encoding="gbk") as f:
        for line in f:
            cleaned_line = line.strip().replace('（', '(').replace('）', ')').replace('！', '!').replace('？', '?').replace('，', ',').replace('；', ';').replace('：', ':').replace('”', '"').replace('“', '"').replace("’", "'").replace("‘", "'").replace("—", "-")
            cleaned_corpus += cleaned_line
    return cleaned_corpus

def new_list(corpus):
    sep = 0
    start = 0
    new_text = []
    corpus_list = list(corpus)
    for i in range(len(corpus)):
        if corpus_list[i] in '~`!@#$%^&*()_+-=<>?:"{}|,./;\'[]\\':
            if sep < 1:
                new_text.append(''.join(corpus_list[start:i+1]) + '<sep>')
                state = i
                start = i + 1
                sep += 1
            else:
                new_text.append(''.join(corpus_list[start:i+1]) + '<eos>')
                start = i + 1
                sep -= 1
        
    new_text = ''.join(new_text)
    with open('/Users/henryzheng/Desktop/modified_corpus.txt', 'w', encoding='utf-8') as file:
        file.write(new_text)
    return new_text

#样本生成
def build_sample(sample_num, vocab, window, target, window_size):
    dataset_x = []
    dataset_y = []
    for i in range(len(window)):
        #window_string = ''.join(window[i])
        x = [vocab.get(word, vocab["<UNK>"]) for word in window[i]]
        if len(x) < window_size:
            x = x + [-1] * (window_size - len(x))
        dataset_x.append(x)
        #print(x,'x')
        #print(window_string,'window_string')
    print(len(window),'window')
    for j in range(len(target)):
        #target_string = ''.join(target[j])
        y = [vocab.get(word, vocab["<UNK>"]) for word in target[i]]
        if len(y) < window_size:
            y = y + [-1] * (window_size - len(y))
        dataset_y.append(y)
        #print(y,'y')
        #print(target_string,'target_string')
    print(len(target),'target')
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    
if __name__ == '__main__':
    vocab = build_vocab('vocab.txt')
    corpus = load_and_clean_corpus('corpus.txt')
    new_text = new_list(corpus)
    total_num = 1000
    window_size = 99
    text = new_text[:total_num]
    split_text = text.split('<eos>')
    window = []
    target = []
    search_char = '<sep>'
    for window_text in split_text:
        window.append(window_text)
        target_index = window_text.find(search_char)
        target.append(window_text[target_index + 5:] + '<eos>')
    
    x,y = build_sample(5 , vocab, window, target, window_size)
    print(x)
    print(y)
