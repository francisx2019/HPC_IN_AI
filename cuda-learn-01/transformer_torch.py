import torch
import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), \
        "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # 获取训练的samples size
        N = query.shape[0]
        # q,k,v的维度为(batch_size, seq_len, embed_size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 将(batch_size, seq_len, embed_size) 经过linear层后，维度变为(batch_size, seq_len, heads, head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        # 将 embedding reshape成 self.heads 和 self.head_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # 进行query*keys, queries的维度为(batch_size, query_len, heads, head_dim), 
        # keys的维度为(batch_size, key_len, heads, head_dim)
        # energy的维度为(batch_size, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None: energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # attention的维度为(batch_size, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # attention shape: (batch_size, heads, query_len, key_len)
        # values shape: (batch_size, value_len, heads, head_dim)
        # out shape: (batch_size, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", 
              [attention, values]).reshape(N, query_len, self.embed_size)
        
        return self.fc_out(out) # (batch_size, query_len, embed_size)
        