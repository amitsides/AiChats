import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=2, head_dim=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        cos_sin = self.rotary_emb(x, L)
        cos, sin = cos_sin.real, cos_sin.imag
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.reshape(B, L, -1)
        out = self.o_proj(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x = swish * self.w2(x)
        x = self.w3(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class ImprovedTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        max_position_embeddings=2048,
        dropout=0.1
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                dropout
            ) for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

def main():
    # Hyperparameters
    hyperparameters = {
        "model_name": "improved-transformer",
        "b": 32,  # batch size
        "s": 128,  # sequence length
        "h": 512,  # hidden size
        "n": 8,  # number of attention heads
        "v": 30522,  # vocabulary size
        "p": 4,  # number of partitions (not directly used in this implementation)
        "q": 64,  # SUMMA dimension (not directly used in this implementation)
        "N": 6,  # number of Transformer layers
        "dropout": 0.1,
        "epochs": 10,
        "learning_rate": 0.001,
        "output_format": "safetensors"
    }

    # Create model
    model = ImprovedTransformerModel(
        vocab_size=hyperparameters["v"],
        hidden_size=hyperparameters["h"],
        num_hidden_layers=hyperparameters["N"],
        num_attention_heads=hyperparameters["n"],
        intermediate_size=hyperparameters["h"] * 4,
        max_position_embeddings=hyperparameters["s"],
        dropout=hyperparameters["dropout"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load and preprocess dataset
    dataset = load_dataset("imdb")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=hyperparameters["s"])
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=hyperparameters["epochs"],
        per_device_train_batch_size=hyperparameters["b"],
        per_device_eval_batch_size=hyperparameters["b"],
        learning_rate=hyperparameters["learning_rate"],
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    main()