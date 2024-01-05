import torch
import torch.nn as nn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=16, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = self.dim // self.heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(self.dim, bias=False)
        self.nonlinearity = nn.SiLU()

        self.fc_in = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.LayerNorm(self.dim, bias=False),
            nn.ReLU()
        )
        
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        self.fc_out = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, y=None, attention_mask=None, is_causal=True):
        y = x if y is None else y
        
        x_shape = x.shape
        y_shape = y.shape

        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.fc_in(x)

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = q.reshape(x_shape[0], x_shape[1], self.heads, x_shape[2] // self.heads).transpose(1, 2)
        k = k.reshape(y_shape[0], y_shape[1], self.heads, y_shape[2] // self.heads).transpose(1, 2)
        v = v.reshape(y_shape[0], y_shape[1], self.heads, y_shape[2] // self.heads).transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout, is_causal=is_causal)
        x = x.transpose(1, 2).contiguous().view(x_shape[0], x_shape[1], x_shape[2])

        x = self.fc_out(x)

        return x

class GPTBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(GPTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.attn = MultiHeadAttention(embed_size, heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

    def forward(self, x, y=None, is_causal=True):
        attn_output = self.attn(self.norm1(x), y=y, attention_mask=None, is_causal=is_causal)
        x = x + attn_output

        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x
    
class AudioGPT(nn.Module):
    def __init__(self, vocab_size, n_codebooks, codebook_size, embed_size, num_layers, heads, forward_expansion, dropout, text_max_length, codebook_max_length):
        super(AudioGPT, self).__init__()

        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.text_max_length = text_max_length
        self.codebook_max_length = codebook_max_length

        self.text_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, self.embed_size) for _ in range(self.n_codebooks)]
        )
        self.text_positional_embeddings = nn.ModuleList(
            [nn.Embedding(self.text_max_length, self.embed_size) for _ in range(self.n_codebooks)]
        )

        self.codebook_embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.embed_size) for _ in range(self.n_codebooks)]
        )
        self.codebook_positional_embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_max_length, self.embed_size) for _ in range(self.n_codebooks)]
        )

        self.embedding_dropout = nn.Dropout(self.dropout)

        self.codebook_self_attentions = nn.ModuleList(
            [nn.ModuleList([GPTBlock(self.embed_size, self.heads, self.dropout, self.forward_expansion) for _ in range(self.num_layers)]) 
             for _ in range(self.n_codebooks)]
        )
        self.codebook_cross_attentions = nn.ModuleList(
            [nn.ModuleList([GPTBlock(self.embed_size, self.heads, self.dropout, self.forward_expansion) for _ in range(self.num_layers)]) 
             for _ in range(self.n_codebooks)]
        )

        self.codebook_heads = nn.ModuleList([nn.Linear(self.embed_size, self.codebook_size) for _ in range(self.n_codebooks)])

    def forward(self, codes, text):
        _, _, seq_length = codes.shape

        codebook_outputs = []
        for i in range(self.n_codebooks):
            embedded_texts = self.text_embeddings[i](text)
            embedded_texts = self.embedding_dropout(embedded_texts)

            text_position_ids = torch.arange(text.size(1), dtype=torch.long, device=text.device)
            text_position_embeddings = self.text_positional_embeddings[i](text_position_ids)

            embedded_texts += text_position_embeddings

            embedded_codes = self.codebook_embeddings[i](codes[:, i, :])
            embedded_codes = self.embedding_dropout(embedded_codes)

            codebook_position_ids = torch.arange(seq_length, dtype=torch.long, device=codes.device)
            codebook_position_embeddings = self.codebook_positional_embeddings[i](codebook_position_ids)
            
            embedded_codes += codebook_position_embeddings

            for j in range(self.num_layers):
                embedded_codes = self.codebook_self_attentions[i][j](embedded_codes, y=None, is_causal=True)

            for j in range(self.num_layers):
                embedded_codes = self.codebook_cross_attentions[i][j](embedded_codes, y=embedded_texts, is_causal=False)
            
            output_for_codebook = self.codebook_heads[i](embedded_codes)
            output_for_codebook = output_for_codebook.view(-1, 1, output_for_codebook.shape[1], self.codebook_size).contiguous()
            codebook_outputs.append(output_for_codebook)

        output = torch.cat(codebook_outputs, dim=1)

        return output