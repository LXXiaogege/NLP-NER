import torch
from torch import nn
from torchcrf import CRF


class bilstm_crf(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim, lstm_hidden_size, lstm_num_layers=1):
        super(bilstm_crf, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden_size // 2, num_layers=lstm_num_layers,
                              batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=lstm_hidden_size, out_features=num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs):
        x, mask = inputs  # x : batch,max_len
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
#     model((x,mask))
