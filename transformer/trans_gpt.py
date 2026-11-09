import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_dim,
            max_seq_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

    def forward(self,input_ids):

        B,L = input_ids.shape

        positions = torch.arange(
            L,
            device = input_ids.device
        ).unsqueeze(0)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        return x

class GPTDecoderBlock(nn.Module):

    def __init__(self,embed_dim,num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first= True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4,embed_dim)
        )

    def forward(self,x):

        B,L,D = x.shape

        causal_mask = torch.triu(
                torch.ones(
                    L,L,
                    device = x.device,
                    dtype= torch.bool
                ),
                diagonal= 1
        )
        norm_x = self.norm1(x)

        attn_output, _ = self.attention(
        norm_x,
        norm_x,
        norm_x,
        attn_mask=causal_mask,
        need_weights=False
        )

        x = x + attn_output
        x = x + self.ffn(self.norm2(x))

        return x
class GPTModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_dim,
            num_heads,
            num_layers,
            max_seq_len
    ):
        super().__init__()

        self.embedding = GPTEmbedding(
            vocab_size,
            embed_dim,
            max_seq_len
        )
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                embed_dim,
                num_heads
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,

        )
    def forward(self,input_ids):

        x =self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
if __name__ == "__main__":

    vocab_size = 10000
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    max_seq_len = 512

    model = GPTModel(
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len
    )

    input_ids = torch.randint(
        0,
        vocab_size,
        (4, 10)
    )

    logits = model(input_ids)

    print("Input:", input_ids.shape)
    print("Logits:", logits.shape)

    print("Decoder Blocks:", len(model.blocks))