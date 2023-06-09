import pickle
import torch
import os
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定使用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(use_bert=False):
    if use_bert:
        model = torch.load("../checkpoint/bert-bi_lstm-crf.pt").to(device)
    else:
        model = torch.load("../checkpoint/bi_lstm-crf.pt").to(device)
    with open('../data/MSRA/preprocessed/MSRA.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)

    sentence = "美方有哈佛大学典礼官亨特、美国驻华大使尚慕杰等。"

    ids = []
    max_len = 50
    if use_bert:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        ids = tokenizer.encode(sentence, padding='max_length', truncation=True,
                               max_length=max_len, return_tensors='pt')[0]
        ids = ids.tolist()
    else:
        for word in sentence:
            if word in word2id:
                ids.append(word2id[word])
            else:
                ids.append(word2id['unknow'])
        # 填充和截断
        ids_len = len(ids)
        if ids_len >= max_len:
            ids = ids[:max_len]
        else:
            ids.extend((max_len - ids_len) * [0])

    ids = torch.tensor(ids, dtype=torch.long).view((1, max_len))
    mask = torch.tensor(ids).bool()
    crf_input, out = model((ids.to(device), mask.to(device)))
    labels = [id2tag[i] for i in out[0]]

    result = []
    for i, j in zip(sentence, labels):
        result.append(i + '/' + j)
    print(result)


if __name__ == '__main__':
    inference(use_bert=True)
