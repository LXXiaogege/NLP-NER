import os
import pickle
from models.BiLSTM_CRF import bilstm_crf
from torch.utils.data import DataLoader
from data.MSRA.MSRA_dataset import MSRA
from torch.optim import Adam
from torch import nn
from torchcrf import CRF
import torch
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 初始化模型
vocab_size = len(word2id) + 1
num_labels = len(tag2id)
embed_dim = 100
lstm_hidden_size = 200
model = bilstm_crf(vocab_size=vocab_size, num_labels=num_labels, embed_dim=embed_dim,
                   lstm_hidden_size=lstm_hidden_size).to(device)
# 加载数据集
train_dataset = MSRA(x_train, y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_dataset = MSRA(x_valid, y_valid)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, drop_last=True)
test_dataset = MSRA(x_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, drop_last=True)

# 定义训练参数
epochs = 10
lr = 0.001
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)
crf = CRF(num_labels, batch_first=True).to(device)

loss_print_nums = 100
for i in range(epochs):
    print('epochs is : ', i)
    model.train()
    for idx, (x, y) in enumerate(train_dataloader):
        mask = y.bool()
        crf_input, out = model((x.to(device), mask.to(device)))
        loss = -crf(crf_input, y.to(device), mask.to(device), reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % loss_print_nums == 0:
            print('batch: ' + str(idx + 1) + ' loss: ', loss.item())

    # val evaluate for every epoch
    print("epoch " + str(i) + " evaluate")
    model.eval()
    pred_list, true_list = [], []
    for idx, (x, y) in enumerate(val_dataloader):
        mask = y.bool()
        crf_input, out = model((x.to(device), mask.to(device)))
        tag_seq = crf.decode(crf_input.to(device), mask.to(device))
        for pred_tag, true_tag in zip(tag_seq, y):
            true_tag = true_tag.tolist()[:len(pred_tag)]
            pred_list.extend(pred_tag)
            true_list.extend(true_tag)

    target_names = ['B_ns', 'B_nr', 'B_nt', 'M_nt', 'M_nr', 'M_ns', 'E_nt', 'E_nr', 'E_ns', 'o']
    report = classification_report(y_true=true_list, y_pred=pred_list, target_names=target_names)
    print(report)

print("save model")
torch.save(model, '../checkpoint/bi_lstm-crf.pt')  # 保存整个模型
print("training success")
