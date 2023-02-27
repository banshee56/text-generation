from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, zeros


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size  # m
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size    # n
        self.num_layers = num_layers    # l

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        self.rnn = nn.ModuleList()
        for i in range(num_layers):
            if rnn_type == "basic_rnn":
                self.rnn.append(BasicRNNCell(vocab_size, hidden_size))
            elif rnn_type == "lstm_rnn":
                self.rnn.append(LSTMCell(vocab_size, hidden_size))
            else:
                raise ValueError(f"Unknown RNN type {rnn_type}")
            vocab_size = hidden_size
        

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x is the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """
        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence
        outs = zeros((x.shape[0], x.shape[1], h.shape[2]))   # shape (B x T x m)

        # for each time step, we start with one input
        for t in range(x.shape[1]):
            x_in = x[:, t, :]           # input volume x_t at time step t, shape (B x n)
            new_h = zeros((h.shape))    # store hidden states computed at this time step, shape (l x B x m)

            for l in range(self.num_layers):
                if self.rnn_type == "basic_rnn":
                    # first parameter: the activation from the previous layer, shape (B x n)
                    # second parameter: the hidden state from the previous time step, shape (B x m)
                    x_out = self.rnn[l].forward(x_in, h[l])                   # x_out shape (B x m)

                elif self.rnn_type == "lstm_rnn":
                    x_out, c = self.rnn[l].forward(x_in, h[l], c[l])

                else:
                    raise ValueError(f"Unknown RNN type {self.rnn_type}")

                # after finisihing computing activation for layer l
                # we can use this as the hidden state input for the next layer l+1
                x_in = x_out                            # shape should be (B x n)
                new_h[l] = x_out                        # shape (B x m)
                # change c!!!

            # after computing all the activations for time step t
            # we can compute the information from the current time step to be used for the next time step
            h = new_h                                   # update h to be the hidden state of the current time step
            outs[:, t, :] = x_out                       # add this time step's final layer's x_out, hidden state, (B x m)

        return outs, h, c
