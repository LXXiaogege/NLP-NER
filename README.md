# 命名实体识别（Named entity recognition）

> 基于深度学习的命名实体识别任务学习。目前有BiLstm-CRF和BERT-BiLstm-CRF模型


One to two paragraph statement about your product and what it does.

![](header.png)

## Installation

```sh
pip install -r requirements.txt
```

## Usage example

BiLstm-CRF和BERT-BiLstm-CRF模型在数据预处理的tokenizer不一样，两个模型不能用一个数据进行训练，在使用时应指定use_bert=True/False,再执行程序

```sh
python ../data/MSRA/data_preprocessing.py
```

```sh
python ./training/bilstm_crf_training.py
```

