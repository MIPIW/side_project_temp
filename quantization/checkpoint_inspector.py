from base_modules import *
from transformers import AutoTokenizer, AutoModelForCausalLM


def initialize_model(hyperparameters):
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



if __name__ == "__main__":

    model, tokenizer = initialize_model(None)
    
    state_dict = torch.load("./outputs/checkpoint-3/model.safetensors", map_location="cpu")    
    load_result = model.load_state_dict(state_dict, False)
    del state_dict

    print([name for name in model.named_parameters()])