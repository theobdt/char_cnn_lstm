import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Input shape: batch_size x word_length x char_embedding_size"""

    def __init__(self, filters_width, num_filters, char_embedding_size):
        super().__init__()
        assert len(filters_width) == len(
            num_filters
        ), "filters_width and num_layers different size"

        self.filters_width = filters_width
        self.num_filters = num_filters
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(char_embedding_size, f, w)
                for w, f in zip(filters_width, num_filters)
            ]
        )
        self.tanh = nn.Tanh()

    def forward(self, x, debug=False):

        if debug:
            print("conv input")
            print(x.shape)
        device = self.convs[0].weight.device
        result = torch.zeros(
            (x.shape[0], sum(self.num_filters)), device=device
        )
        current_idx = 0
        for i, conv in enumerate(self.convs):
            current_size = self.num_filters[i]
            y = self.tanh(conv(x))
            if debug:
                print("conv out")
                print(y.shape)
            m = torch.max(y, dim=2)[0]
            if debug:
                print("max out")
                print(m.shape)
            result[:, current_idx : current_idx + current_size] = m
            current_idx += current_size
        if debug:
            print("shape result")
            print(result.shape)

        return result


class HighwayNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.T = nn.Linear(input_size, 1)
        self.H = nn.Linear(input_size, input_size)

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        result = t * F.relu(self.H(x)) + (1 - t) * x
        return result


class CharCNNLSTM(nn.Module):
    """Input shape: batch_size x seq_len x word_length"""

    def __init__(
        self,
        char_vocab_size,
        char_embedding_size,
        word_vocab_size,
        filters_width,
        num_filters,
        num_layers,
        hidden_size,
        dropout,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.char_embeddings = nn.Embedding(
            char_vocab_size, char_embedding_size
        )
        self.char_cnn = CharCNN(
            filters_width, num_filters, char_embedding_size
        )
        self.highway_net = HighwayNet(sum(num_filters))
        self.lstm = nn.LSTM(
            sum(num_filters),
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.drop_layer = nn.Dropout(dropout)
        self.hidden2words = nn.Linear(hidden_size, word_vocab_size)

    def forward(self, x, hidden, debug=False):
        # shape : (batch_size, seq_len, word_length)
        initial_shape = x.shape
        if debug:
            print("input")
            print(x.shape)
        x = x.view(-1, x.shape[2])
        if debug:
            print("reshaped")
            print(x.shape)
        # shape : (batch_size * seq_len, word_length)

        x = self.char_embeddings(x)
        # shape : (batch_size * seq_len, word_length, embedding_size)
        if debug:
            print("after emb")
            print(x.shape)

        x = x.transpose(1, 2)
        # shape : (batch_size * seq_len, embedding_size, word_length)
        if debug:
            print("after trans")
            print(x.shape)

        x = self.char_cnn(x)
        # shape : (batch_size * seq_len, sum(num_filters))
        if debug:
            print("after conv")
            print(x.shape)

        x = self.highway_net(x)
        if debug:
            print("after highway")
            print(x.shape)

        x = x.view(initial_shape[0], initial_shape[1], -1)
        # shape : (batch_size, seq_len, sum(num_filters)
        if debug:
            print("after final reshape")
            print(x.shape)

        outputs, hidden = self.lstm(x, hidden)
        # output shape : (batch_size, seq_len, hidden_size)
        if debug:
            print("after lstm")
            print(x.shape)
        outputs = self.drop_layer(outputs)

        outputs = self.hidden2words(outputs)
        if debug:
            print("after linear")
            print(outputs.shape)
        return outputs, hidden

    def init_hidden(self, batch_size):
        weights = next(self.parameters())
        # to have the same dtype as model's parameters
        return (
            weights.new_zeros(self.num_layers, batch_size, self.hidden_size),
            weights.new_zeros(self.num_layers, batch_size, self.hidden_size),
        )
