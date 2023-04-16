import codecs
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from transformers import AutoTokenizer

"""
中文分词，基于字标注：
B | 词首
M | 词中
E | 词尾
O | 单字
B 表示一个词的开始，E 表示一个词的结尾，M 表示词中间的字。
"""

tag2id = {'': 0,  # padding字对应的tag : 0
          'B_ns': 1, 'B_nr': 2, 'B_nt': 3, 'M_nt': 4, 'M_nr': 5, 'M_ns': 6, 'E_nt': 7, 'E_nr': 8, 'E_ns': 9, 'o': 10}

id2tag = {0: '', 1: 'B_ns', 2: 'B_nr', 3: 'B_nt', 4: 'M_nt', 5: 'M_nr', 6: 'M_ns', 7: 'E_nt', 8: 'E_nr', 9: 'E_ns',
          10: 'o'}

use_bert = True
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def wordtag():
    """
    对MSRA数据集做字标注
    :return:
    """
    input_data = codecs.open('train1.txt', 'r', 'utf-8')
    output_data = codecs.open('preprocessed/wordtag.txt', 'w', 'utf-8')
    for line in input_data.readlines():
        # line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
        line = line.strip().split()

        if len(line) == 0:
            continue
        for word in line:
            word = word.split('/')
            if word[1] != 'o':  # 对命名实体部分做字标注
                if len(word[0]) == 1:  # 单个字 B
                    output_data.write(word[0] + "/B_" + word[1] + " ")
                elif len(word[0]) == 2:  # 两个字 BE
                    output_data.write(word[0][0] + "/B_" + word[1] + " ")
                    output_data.write(word[0][1] + "/E_" + word[1] + " ")
                else:  # 两个字以上 BM...E
                    output_data.write(word[0][0] + "/B_" + word[1] + " ")
                    for j in word[0][1:len(word[0]) - 1]:
                        output_data.write(j + "/M_" + word[1] + " ")
                    output_data.write(word[0][-1] + "/E_" + word[1] + " ")
            else:  # 'o' 的意思是不是命名实体实体
                for j in word[0]:
                    output_data.write(j + "/o" + " ")
            pass
        output_data.write('\n')

    input_data.close()
    output_data.close()


def construct_vocab():
    """
    制作vocabulary，编码数据
    :return:
    """
    datas = list()
    labels = list()
    input_data = codecs.open('preprocessed/wordtag.txt', 'r', 'utf-8')
    for line in input_data.readlines():
        line = re.split('[，。；！：？、‘’“”]/[o]'.encode('utf-8').decode('utf-8'), line.strip())  # 根据标点符号分句
        for sen in line:
            sen = sen.strip().split()
            if len(sen) == 0:
                continue
            linedata = []
            linelabel = []
            num_not_o = 0  # 实体字个数
            for word in sen:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(tag2id[word[1]])  # encode label

                if word[1] != 'o':
                    num_not_o += 1
            if num_not_o != 0:
                datas.append(linedata)
                labels.append(linelabel)

    input_data.close()

    def flat2gen(alist):
        for item in alist:
            if isinstance(item, list):
                for subitem in item: yield subitem
            else:
                yield item

    all_words = list(flat2gen(datas))
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()  # 统计每个词出现的次数，相当于去重
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)  # 给每一个字一个id映射，注意这里是从1开始，因为我们填充序列时(padding)使用0填充的，也就是id为0的已经被占用了
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    # 加入一个unknow，如果没出现的字就用unknow的id代替
    word2id["unknow"] = len(word2id) + 1
    id2word[len(id2word) + 1] = "unknow"

    return datas, labels, word2id, id2word


def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    if use_bert:
        sentence = ''
        for i in words:
            sentence = sentence + i
        ids = tokenizer.encode(sentence, padding='max_length', truncation=True, max_length=50, return_tensors='pt')[0]
        ids = ids.tolist()
    else:
        ids = list(word2id[words])
        if len(ids) >= max_len:  # 长则弃掉
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


def y_padding(ids):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


if __name__ == '__main__':
    # 对每个字进行标注
    wordtag()

    # 构建词表，
    datas, labels, word2id, id2word = construct_vocab()

    # padding
    max_len = 50
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    # 划分数据集

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    with open(os.path.join('preprocessed/MSRA.pkl'), 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)
    print('** Finished saving the data.')
