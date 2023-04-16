import torch
from torch import nn
from torchcrf import CRF
from transformers import BertModel


class bilstm_crf(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim, lstm_hidden_size, lstm_num_layers=1, use_bert=False):
        super(bilstm_crf, self).__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_bert = use_bert

        if self.use_bert:
            self.bert = BertModel.from_pretrained("bert-base-chinese")
            self.embed_dim = 768
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.bilstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_size // 2,
                              num_layers=self.lstm_num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, inputs):
        x, mask = inputs  # x : batch,max_len
        if self.use_bert:
            with torch.no_grad():
                embed = self.bert(x)['last_hidden_state']
        else:
            embed = self.embedding(x)
        lstm_out, (h_n, c_n) = self.bilstm(embed)
        crf_input = self.linear(lstm_out)  # 将lstm输出映射到标签空间
        out = self.crf.decode(crf_input, mask)
        return crf_input, out

# if __name__ == '__main__':
#     x = torch.zeros(size=(32, 50), dtype=torch.long)
#     y = torch.zeros(size=(32, 50), dtype=torch.long)
#     mask = y.bool()
#     vocab_size = 100
#     num_labels = 5
#     embed_dim = 100
#     lstm_hidden_size = 200
#     model = bilstm_crf(vocab_size=vocab_size, num_labels=num_labels, embed_dim=embed_dim,
#                        lstm_hidden_size=lstm_hidden_size)
#     model((x, mask))  # crf.decode有错误, y不能初始化为zeros
