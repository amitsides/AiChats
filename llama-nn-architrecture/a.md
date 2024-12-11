This implementation includes all the key architectural components we discussed:

RMSNorm instead of LayerNorm
Rotary Positional Embeddings (RoPE)
Grouped-Query Attention (GQA)
SwiGLU activation in the feed-forward network
No bias terms in linear layers
Pre-normalization architecture

Key features of the implementation:

Modular design with separate classes for each component
Full implementation of rotary embeddings
Grouped-query attention mechanism
SwiGLU activation in feed-forward networks
Proper scaling and normalization

To use this model, you would do something like:

model = create_small_llama()
batch_size = 4
seq_length = 128
input_ids = torch.randint(0, 32000, (batch_size, seq_length))
output = model(input_ids)

