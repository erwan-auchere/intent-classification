import torch

def permute(tensor):
    """
    Permutes all dimensions of a tensor.

    Input : tensor of dimension (d1, d2, ... dn)
    Output : tensor of dimension (dn, ..., d2, d1)
    """
    n = len(tensor.shape)
    return torch.permute(tensor, list(range(n-1, -1, -1)))

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
            hidden_input = permute(permute(1-D[t,:])*permute(h[t-1,...])) if t>0 else torch.zeros(batch_size, self.hidden_size)
            h[t,...] = self.forward_cell(X[t,...], hidden_input)
        

        ## Second direction
        hb = torch.zeros(sequence_len, batch_size, self.hidden_size) # h contains the hidden states
        for t in reversed(range(sequence_len)):
            # We reset the hidden input if the speaker changes
            hidden_input = permute(permute(1-D[t+1,:])*permute(hb[t+1,...])) if t<sequence_len-1 else torch.zeros(batch_size, self.hidden_size)
            hb[t,...] = self.backward_cell(X[t,...], hidden_input)
        
        # Output has shape (sequence_len, batch_size, 2*hidden_size)
        return torch.cat([h, hb], dim=2)


class HierarchicalEncoder(torch.nn.Module):
    def __init__(self, input_size, sequence_length=5, hidden_size=128, persona_level=True):
        super(HierarchicalEncoder, self).__init__()
        self.sequence_length = sequence_length

        self.word_level = torch.nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        
        if persona_level:
            self.persona_level = DiscontinuedGRU(2*hidden_size, hidden_size)
        else:
            self.persona_level = None

        self.sentence_level = torch.nn.GRU(2*hidden_size, hidden_size, bidirectional=True)
    

    def forward(self, X, P):
        # X is a tensor shape (batch_size, sequence_length, max_sentence_length, input_size), where each word is represented by a vector of size input_size
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
        if self.persona_level is not None:
            hp = self.persona_level(hw, P.transpose(0, 1))
        else:
            hp = hw

        ## Sentence-level encoding
        hs = self.sentence_level(hp)[0] # shape (sequence_length, batch_size, 2*hidden_size)
        Hi = hs[-1, ...] # last timestep: shape (batch_size, 2*hidden_size)

        return hs, Hi


class SoftGuidedAttentionDecoder(torch.nn.Module):
    def __init__(self, hidden_size=128, sequence_length=5):
        super(SoftGuidedAttentionDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.attention = torch.nn.Linear(4*hidden_size, 1)
        self.recurrent_cell = torch.nn.GRUCell(2*hidden_size, 2*hidden_size)

    def forward(self, X):
        ## X is the output of the encoder
        ## Its shape is (sequence_length, batch_size, 2*hidden_size)
        
        output = torch.zeros(*X.shape) # (sequence_length, batch_size, 2*hidden_size)
        hidden_state = X[-1,...].clone() # (batch_size, 2*hidden_size)

        for t in range(self.sequence_length): # (batch_size, sequence_length)
            #  "h^d_{t-1}" is the same for all j so we repeat is along the corresponding axis
            expanded_hidden_state = torch.stack([hidden_state for _ in range(self.sequence_length)], dim=0) # (sequence_length, batch_size, 2*hidden_size)
            attention_layer_input = torch.cat([expanded_hidden_state, X], dim=-1) # (sequence_length, batch_size, 4*hidden_state)

            attention_weights = self.attention(attention_layer_input).squeeze(dim=-1) # (sequence_length, batch_size) -> a[j,n] correspond à alpha_{t,j} pour la n-ième observation
            attention_weights += torch.Tensor([ int(j==t) for j in range(self.sequence_length)]).repeat(X.shape[1], 1).transpose(0,1)
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=0)

            context_vectors = ( permute(permute(X)*permute(attention_weights)) ).sum(dim=0) # (batch_size, 2*hidden_size) // mutliplication is broadcast along third axis, and dimension 0 (sequence_length) is removed by the sum
            output[t,...] = self.recurrent_cell(context_vectors, hidden_state) # (batch_size, 2*hidden_size)

            hidden_state = output[t,...].clone()
        
        # shape (sequence_length, batch_size, 2*hidden_size)
        return output


class HardGuidedAttentionDecoder(torch.nn.Module):
    def __init__(self, hidden_size=128, sequence_length=5):
        super(HardGuidedAttentionDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.recurrent_cell = torch.nn.GRUCell(2*hidden_size, 2*hidden_size)

    def forward(self, X):
        ## X is the output of the encoder
        ## Its shape is (sequence_length, batch_size, 2*hidden_size)
        
        output = torch.zeros(*X.shape) # (sequence_length, batch_size, 2*hidden_size)
        hidden_state = X[-1,...].clone() # (batch_size, 2*hidden_size)

        for t in range(self.sequence_length): # (batch_size, sequence_length)
            context_vectors = X[t,...].clone()
            output[t,...] = self.recurrent_cell(context_vectors, hidden_state)
            hidden_state = output[t,...].clone()
        
        # shape (sequence_length, batch_size, 2*hidden_size)
        return output
    
class VanillaAttentionDecoder(torch.nn.Module):
    def __init__(self, hidden_size=128, sequence_length=5):
        super(VanillaAttentionDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.attention = torch.nn.Linear(4*hidden_size, 1)
        self.recurrent_cell = torch.nn.GRUCell(2*hidden_size, 2*hidden_size)

    def forward(self, X):
        ## X is the output of the encoder
        ## Its shape is (sequence_length, batch_size, 2*hidden_size)
        
        output = torch.zeros(*X.shape) # (sequence_length, batch_size, 2*hidden_size)
        hidden_state = X[-1,...].clone() # (batch_size, 2*hidden_size)

        for t in range(self.sequence_length): # (batch_size, sequence_length)
            #  "h^d_{t-1}" is the same for all j so we repeat is along the corresponding axis
            expanded_hidden_state = torch.stack([hidden_state for _ in range(self.sequence_length)], dim=0) # (sequence_length, batch_size, 2*hidden_size)
            attention_layer_input = torch.cat([expanded_hidden_state, X], dim=-1) # (sequence_length, batch_size, 4*hidden_state)

            attention_weights = self.attention(attention_layer_input).squeeze(dim=-1) # (sequence_length, batch_size) -> a[j,n] correspond à alpha_{t,j} pour la n-ième observation
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=0)
            #attention_weights += torch.Tensor([ int(j==t) for j in range(self.sequence_length)]).repeat(X.shape[1], 1).transpose(0,1)
            #attention_weights = torch.nn.functional.softmax(attention_weights, dim=0)

            context_vectors = ( permute(permute(X)*permute(attention_weights)) ).sum(dim=0) # (batch_size, 2*hidden_size) // mutliplication is broadcast along third axis, and dimension 0 (sequence_length) is removed by the sum
            output[t,...] = self.recurrent_cell(context_vectors, hidden_state) # (batch_size, 2*hidden_size)

            hidden_state = output[t,...].clone()
        
        # shape (sequence_length, batch_size, 2*hidden_size)
        return output




class Seq2SeqModel(torch.nn.Module):
    # Embedding, encoder, decoder, linear+softmax

    def __init__(self, pretrained_embeddings, encoder, decoder, nb_classes, hidden_size=128):
        super(Seq2SeqModel, self).__init__()

        self.embedder = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.encoder = encoder
        self.decoder = decoder
        self.output = torch.nn.Linear(2*hidden_size, nb_classes)

    def forward(self, X):
        # X has shape (batch_size, sequence_length, max_sentence_length)
        # For the dataset that we use, the speaker changes at every new utterance so P is filled with 1s
        # For a different dataset P should be an input of the model
        P = torch.ones(*X.shape[:-1])
        # P has shape (batch_size, sequence_length)
        embedded = self.embedder(X) # (batch_size, sequence_length, max_sentence_length, embedding_dim)
        encoded = self.encoder(embedded, P)[0] # (sequence_length, batch_size, 2*hidden_size)
        decoded = self.decoder(encoded) # (sequence_length, batch_size, 2*hidden_size)
        scores = self.output(decoded) # (sequence_length, batch_size, nb_classes)
        probas = torch.nn.functional.softmax(scores, dim=-1) # (sequence_length, batch_size, nb_classes)
        return probas.transpose(0, 1) # (batch_size, sequence_length, nb_classes)


