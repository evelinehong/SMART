'''
group-head attention with the whole attention-->add&norm-->feedforward-->add&norm block
dropout means the dropout rate in residual connection
N means number of blocks,setting to 2
'''
import torch.nn as nn

from .baseRNN import BaseRNN
from .attention_1 import GroupAttention
from test import PositionwiseFeedForward
from copy import deepcopy
from transformer import Encoder,EncoderLayer


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embed_model=None, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 rnn_cell=None, rnn_cell_name='gru', variable_lengths=True,d_ff=2048,dropout=0.3,N=1):
        super(EncoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p, n_layers, rnn_cell_name)
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        if bidirectional:
            self.d_model = 2*hidden_size
        else:self.d_model = hidden_size
        ff = PositionwiseFeedForward(self.d_model, d_ff, dropout)
        if embed_model is None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model
        if rnn_cell is None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = rnn_cell
        self.group_attention = GroupAttention(8,self.d_model)
        self.onelayer = Encoder(EncoderLayer(self.d_model,deepcopy(self.group_attention),deepcopy(ff),dropout),N)

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        src_mask = self.group_attention.get_mask(input_var)
        output = self.onelayer(output,src_mask)
        return output, hidden
