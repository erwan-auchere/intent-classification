import torch


class DiscontinuedGRU(torch.nn.Module):
    """
    Bidirectional GRU layer with discontinuities.
    Used for the persona-level encoding: when the speaker changes, the hidden state is reinitialized.
    """
    def __init__(self, input_size, hidden_size):
        super(DiscontinuedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.forward_cell = torch.nn.GRUCell(input_size, hidden_size)
        self.backward_cell = torch.nn.GRUCell(input_size, hidden_size)
    
    def forward(self, X, D):
        """
        X : tensor of shape (sequence_len, batch_size, input_size) containing the data.
        D : binary tensor of shape (sequence_len, batch_size) indicating when the hidden state should be reset.
        In our case, D_it = 0 if u_it has the same speaker as u_i(t-1) and 1 if not
        """
        sequence_len, batch_size, _ = X.shape

        ## First direction
        h = torch.zeros(sequence_len, batch_size, self.hidden_size) # h contains the hidden states
        for t in range(sequence_len):
            # We reset the hidden input if the speaker changes
            hidden_input = (1-D[t,:])*h[t-1,...] if t>0 else torch.zeros(batch_size, self.hidden_size)
            h[t,...] = self.forward_cell(X[t,...], hidden_input)
        

        ## Second direction
        hb = torch.zeros(sequence_len, batch_size, self.hidden_size) # h contains the hidden states
        for t in reversed(range(sequence_len)):
            # We reset the hidden input if the speaker changes
            hidden_input = (1-D[t+1,:])*hb[t+1,...] if t<sequence_len-1 else torch.zeros(batch_size, self.hidden_size)
            hb[t,...] = self.backward_cell(X[t,...], hidden_input)
        
        # Output has shape (sequence_len, batch_size, 2*hidden_size)
        return torch.cat([h, hb], dim=2)


class HierarchicalEncoder(torch.nn.Module):
    def __init__(self, input_size, sequence_length=5, hidden_size=128):
        super(HierarchicalEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.word_level = torch.nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.persona_level = DiscontinuedGRU(2*hidden_size, hidden_size)
        self.sentence_level = torch.nn.GRU(2*hidden_size, hidden_size, bidirectional=True)
    

    def forward(self, X, P):
        # X is a tensor shape (batch_size, sequence_length, max_sentence_length, input_size), where each word is represented by a one-hot vector of size input_size
        # P is a binary tensor of shape (batch_size, sequence_length) indicating when the speaker changes

        ## Word-level encoding
        hw = []
        for t in range(self.sequence_length):
            last_output = self.word_level(X[:,t,...])[0][:,-1,:]
            # shape = (batch_size, 2*hidden_size)
            hw.append(last_output)

        hw = torch.stack(hw, dim=0)
        # shape = (sequence_length, batch_size, 2*hidden_size)


        ## Persona-level encoding
        hp = self.persona_level(hw, P.transpose(0, 1))

        ## Sentence-level encoding
        hs = self.sentence_level(hp)[0] # shape (sequence_length, batch_size, 2*hidden_size)
        Hi = hs[-1, ...] # last timestep: shape (batch_size, 2*hidden_size)

        return Hi
