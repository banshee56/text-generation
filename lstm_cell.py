from torch import nn, sigmoid, tanh, Tensor
from math import sqrt
import torch

class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables
        n = self.vocab_size
        m = self.hidden_size
        k = sqrt(1/m)
        p = 4*m

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        self.W = nn.parameter.Parameter(torch.empty((n, p)).uniform_(-k, k))

        # V, the hidden state weights matrix has size (m, (4 * m))
        self.V = nn.parameter.Parameter(torch.empty((m, p)).uniform_(-k, k))

        # b, the vector of biases has size (4 * m)
        self.b = nn.parameter.Parameter(torch.empty(p).uniform_(-k, k))

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """
        a = self.b + torch.mm(x, self.W) + torch.mm(h, self.V)
        a_s = a.split(int(self.W.shape[1]/4), 1)        # split the output into shapes of size (n, m)

        # create the gates
        i = sigmoid(a_s[0])
        f = sigmoid(a_s[1])
        o = sigmoid(a_s[2])
        g = tanh(a_s[3])

        c_out = torch.mul(i, g) + torch.mul(f, c)
        h_out = torch.mul(o, tanh(c_out))

        return h_out, c_out


