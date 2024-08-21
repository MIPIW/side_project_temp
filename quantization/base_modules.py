from torch import nn
from torch.nn import functional as F
import torch
import copy, math
import numpy as np

class Transformer(nn.Module):
    def __init__(self, embeddingLayer, decoderLayers, generatorLayer, tokenizer): # encoder_block, decoder_block, encoder_layers, decoder_layers):
        super(Transformer, self).__init__()

        self.embeddingLayer = embeddingLayer
        self.decoderLayers = decoderLayers
        self.generator = generatorLayer
        self.pad_index = tokenizer.pad_token_id

        self.tokenizer = tokenizer

        self.last_hidden_states = None
        self.logits = None

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.size()[0]
        seq_len = input_ids.size()[1]
    
        subsequent_mask = self.make_subsequent_mask(input_ids, input_ids)
        subsequent_mask = subsequent_mask.repeat(batch_size, 1, 1)
        attention_mask = attention_mask.unsqueeze(2).repeat(1,1,seq_len)
        mask = subsequent_mask & attention_mask    

        inputs = self.embeddingLayer(input_ids)
        last_hidden_states = self.decoderLayers(inputs, mask)
        logits = self.generator(last_hidden_states)
        logits = F.log_softmax(logits, dim = -1) 

        self.last_hidden_states = last_hidden_states
        self.logits = logits

        loss_fn = nn.NLLLoss()
        loss = loss_fn(logits.view(-1, len(self.tokenizer)), labels.view(-1))

        print("loss_______:", loss)
        

        return {"loss": loss,
                "last_hidden_states": self.last_hidden_states,
                "logits": self.logits}

        # out = input_text
        # for i in range(self.l_en):
        #     out = self.encoder(out)
        # for i in range(self.l_de):
        #     out = self.decoder(out)

        # return out

    def make_subsequent_mask(self, query, key):
        query_s, key_s = query.size(1), key.size(1)
        tril = np.tril(np.ones((query_s, key_s)), k = 0).astype("uint8") 
        mask = torch.tensor(tril, dtype = torch.bool, requires_grad = False).to(query.device)
        return mask

class Encoder(nn.Module):
    def __init__(self, EncoderBlock, layers):
        super(Encoder, self).__init__()
        self.n_layers = layers
        self.layers = nn.ModuleList([copy.deepcopy(EncoderBlock) for _ in range(self.n_layers)]) # deepcopy should not share parameters

    def forward(self, x, src_mask):
        out = x
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, FFNN, layernorm, dr_rate):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.FFNN = FFNN
        self.residuals = nn.ModuleList([Residual_connection(copy.deepcopy(layernorm), dr_rate) for _ in range(2)])
        # self.residuals = ?

    def forward(self, input_text, input_mask): # in encoder input_mask = None(bidirectional attention)
        out = input_text
        out = self.residuals[0](out, lambda out: self.self_attention(query = out, key = out, value = out, mask = input_mask))
        out = self.residuals[1](out, lambda out: self.FFNN(out))
        return out
    

    
class Generator(nn.Module):
    def __init__(self, tgt_vocab_size, v):
        super(Generator, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.FFNN = nn.Linear(v, tgt_vocab_size)

    def forward(self, text):
        return self.FFNN(text)



class Embedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super(Embedding, self).__init__()
        self.token_embed = token_embed
        self.pos_embed = pos_embed
        

    def forward(self, input_text):
        
        seq_len = input_text.size()[1]

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_text.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_text)
        position_embeddings = self.pos_embed(position_ids)

        return self.token_embed(input_text) + position_embeddings

# class TokenEmbedding(nn.Module):
#     def __init__(self, v, vocab_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, v)
#         self.v = v

#     def forward(self, input_text):
#         return self.embedding(input_text) * math.sqrt(self.v)


# class PositionalEmbedding(nn.Module):
#     def __init__(self, v, max_len, device):
#         super(PositionalEmbedding, self).__init__()
#         self.encoding = torch.zeros(max_len, v)
#         self.encoding.requires_grad = False
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = torch.arange(0, v, 2) * -(math.log(10000.0) / v)
#         self.encoding[:,0::2] = torch.sin(position * div_term)
#         self.encoding[:,1::2] = torch.cos(position * div_term)

#         self.device  = device


#     def forward(self, x):
#         _, seq_len, _ = x.size() # (batch, seq_len)
#         pos_embed = self.encoding[:, :seq_len]
#         print("pos_embed-----------------", pos_embed.size())
#         print("x------------------------", x.size())
#         out = x + pos_embed.to(self.device)

#         return out
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_attention, head, qkv_fc, out_fc):
        super(MultiheadAttention, self).__init__()
        self.d_attention = d_attention # 임의값. 꼭 hidden representation size와 같을 필요는 없음. query_fully_connected_matrix
        self.h = head
        self.q_fc = copy.deepcopy(qkv_fc) # (hidden_vector(v), attention_size(a)) where a = d_model = v)
        self.k_fc = copy.deepcopy(qkv_fc) # (v, a)
        self.v_fc = copy.deepcopy(qkv_fc) # (v, a)
        self.out_fc = out_fc # (a, v)

    def forward(self, query, key, value, *args, mask = None):
        # qkv: (batch(b), seq_len(s), hidden_vector(v))
        # mask = (b, s, s)
        # returns = (b, h, s, a)
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x) # (b, s, a). constructing query vector
            out = out.view(n_batch, -1, self.h, self.d_attention // self.h).transpose(1,2) # (b,h,s,dk = a/h)
            return out

        q = transform(query, self.q_fc) #(b,h,s,dk)
        k = transform(key, self.k_fc)   #(b,h,s,dk)
        v = transform(value, self.v_fc) #(b,h,s,dk)

        out = self.attention(q,k,v,mask) # (b, h, s, dk)
        out = out.transpose(1,2) # (b, s, h, dk)
        out = out.contiguous().view(n_batch, -1, self.d_attention) # (b,s, h*dk=a=v)
        out = self.out_fc(out) # (b,s,v)

        return out

    def attention(self, query, key, value, mask = None):
        # qkv: (b, h, s, dk)
        size_hidden_vector = key.shape[-1]
        score = torch.matmul(query, key.transpose(-2, -1)) # keep (v, h) still, and (s, dk) @ (dk, s) -> (b, h, s, s)
        score = score / math.sqrt(size_hidden_vector) # scaled dot product attention
        
        mask = mask.unsqueeze(1).repeat(1,self.h,1,1) #(b, s, s) -> (b,h,s,s)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        prob = F.softmax(score, dim = -1) # (b, h, s, s), softmax with last s

        out = torch.matmul(prob, value) # (b,h,s,s) * (b,h,s,dk) = (b,h,s,dk) columnwise multiplication

        return out # (b,h,s,dk)


# minor structure: FFNN, Residual Connection, generator
class PositionwiseFFNN(nn.Module):
    def __init__(self, v, a):
        super(PositionwiseFFNN, self).__init__()
        self.fc1 = nn.Linear(v, a)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(a,v)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class Residual_connection(nn.Module):
    def __init__(self, layernorm, dr_rate):
        super(Residual_connection, self).__init__()
        self.layernorm = layernorm
        self.dropout = nn.Dropout(p = dr_rate)
        self.gelu = nn.GELU()

    def forward(self, x, attention):
        out = x
        out = attention(out) # forwarding
        out = self.layernorm(out) # layernorm
        out = self.gelu(out) # activation
        out = self.dropout(out) # dropout
        out = x + out # residual connection

        return out


def initializa_model(hyperparameters):
    ## initializing toy model 
    hidden_states = 768
    seq_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # embedding
    tokenizer_checkpoint = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    tokenizer.pad_token = "[PAD]"

    model_for_embedding = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint)
    model_for_embedding.resize_token_embeddings(len(tokenizer))

    token_embedding = model_for_embedding.transformer.wte
    positional_embedding = model_for_embedding.transformer.wpe
    embedding = Embedding(token_embedding, positional_embedding)

    for params in embedding.parameters():
        params.requires_grad = False
        print(params.requires_grad)

    # tokenEmbedding = TokenEmbedding(hidden_states, len(tokenizer))
    # positionalEmbedding = PositionalEmbedding(hidden_states, seq_len, device)
    # embedding = Embedding(tokenEmbedding, copy.deepcopy(positionalEmbedding))

    # attention layer
    attention_dim = 64
    num_head = 4
    qkv_fc = nn.Linear(in_features=hidden_states, out_features=attention_dim, bias=True)
    out_fc = nn.Linear(in_features=attention_dim, out_features=hidden_states, bias=True)
    attention = MultiheadAttention(attention_dim, num_head, qkv_fc, out_fc)

    # FFNN
    d_ffnn = 1024
    norm_eps = 1e-05
    ffnn = PositionwiseFFNN(hidden_states, d_ffnn)
    layerNorm = nn.LayerNorm(hidden_states, eps = norm_eps)

    # decoder(AutoRegressive model)
    dropout_rate = 0.1
    num_layers = 1
    decoderBlock = EncoderBlock(attention, ffnn, layerNorm, dropout_rate)
    decoder = Encoder(decoderBlock, num_layers)

    # generator
    generator = Generator(len(tokenizer), hidden_states) 

    #transformer
    model = Transformer(embedding, decoder, generator, tokenizer)

    return model, tokenizer