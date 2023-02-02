import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_embedding_dim,
        n_hidden,
        n_layers,
        dropout=0.5,
        rnn_type="gru",
        bidirectional = True,  # can be elman, lstm, gru
    ):
        super(RNNModel, self).__init__()

        if rnn_type == "elman":
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional = False
            )
        # TODO: implement lstm and gru
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional = True
            )

        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                in_embedding_dim,
                n_hidden,
                n_layers,
                dropout=dropout,
                bidirectional = True
            )
        else:
            raise ValueError("rnn_type must be one of elman, lstm, gru")
            
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.n_dir = 2 if bidirectional else 1
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.Linear(n_hidden, vocab_size)
        self.init_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.in_embedder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            model.in_embedder.weight.requires_grad = True
            return model