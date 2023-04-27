import torch
import torch.nn as nn

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        #* num_units： embedding dim
        self.num_units = num_units
        # * 窗口大小
        self.max_relative_position = max_relative_position
        # * positional embedding table
        #* [7,5]
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        #* [tgt_len]
        range_vec_q = torch.arange(length_q)
        #* [src_len]
        range_vec_k = torch.arange(length_k)
        #* [tgt_len, src_len] = [9,8]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = (distance_mat_clipped + self.max_relative_position).cuda()
        final_mat = torch.tensor(final_mat,device='cuda').long()
        #* [tgt_len, src_len, num_units] = [9,8,5]
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_relative_position, device):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads  # d_k, d_v
        self.max_relative_position = max_relative_position

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position).cuda()
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position).cuda()

        self.fc_q = nn.Linear(d_model, d_model).cuda()  # n_heads * head_dim(d_k, d_v)
        self.fc_k = nn.Linear(d_model, d_model).cuda()
        self.fc_v = nn.Linear(d_model, d_model).cuda()
        
        self.fc_o = nn.Linear(d_model, d_model).cuda()
        
        self.dropout = nn.Dropout(dropout).cuda()
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        """
        forward _summary_

        Parameters
        ----------
        Q: [batch, tgt_len, d_model]
        K: [batch, src_len, d_model]
        V: [batch, src_len, d_model]
        mask: [batch, tgt_len, src_len]  
        mask = padding mask + decoder mask
        -------
        Returns:
        attn_value: [batch, tgt_len, d_model]
        mask: [batch, n_head, tgt_len, src_len]
        """
        batch_size = query.shape[0]
        assert key.shape[1] == value.shape[1]
        src_len = key.shape[1]
        tgt_len = query.shape[1]
        #* [batch, tgt_len, n_head*head_dim]
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        #* r_q1: [batch, n_head, tgt_len, d_k]
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #* r_k1: [batch, n_head, src_len, d_k]
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #* attn: [batch, n_head, tgt_len, src_len]
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 
        #* [batch,tgt_len, n_head*head_dim] 
        #* -> [tgt_len, batch, n_head*head_dim]
        #* -> [tgt_len, batch*n_head, head_dim]
        r_q2 = query.permute(1, 0, 2).contiguous().view(tgt_len, batch_size*self.n_heads, self.head_dim)
        #* [tgt_len, src_len, head_dim]
        r_k2 = self.relative_position_k(tgt_len, src_len)
        #* [tgt_len,batch*n_head,src_len] -> [batch*n_head,tgt_len, src_len]
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, tgt_len, src_len)
        #* [batch, n_head,tgt_len,src_len]
        attn = (attn1 + attn2) / self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,n_heads,1,1)
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(torch.softmax(attn, dim = -1))

        #* r_v1: [batch, n_head, src_len, d_v]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #* [batch, n_head, tgt_len, head_dim]
        weight1 = torch.matmul(attn, r_v1)
        #* [tgt_len, src_len, head_dim]
        r_v2 = self.relative_position_v(tgt_len, src_len)
        #* [tgt_len, batch*n_head,src_len]
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size*self.n_heads, src_len)
        #* [tgt_len, batch*n_head, head_dim]
        weight2 = torch.matmul(weight2, r_v2)
        #* [batch, n_head, tgt_len, head_dim]
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, tgt_len, self.head_dim)
        #* [batch, n_head, tgt_len, head_dim]
        x = weight1 + weight2
        #* [batch,tgt_len,n_head,head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()       
        #* [batch, tgt_len, d_model]
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        return x
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 10
    n_heads = 2
    batch_size = 3
    tgt_len = 9
    src_len = 8
    max_relative_position = 3
    model = MultiHeadAttentionLayer(d_model = d_model, n_heads = n_heads, dropout = 0.1, max_relative_position=max_relative_position, device = device).cuda()
    print(model)
    q = torch.randn(batch_size, tgt_len, d_model).cuda()
    k = torch.randn(batch_size, src_len, d_model).cuda()
    v = torch.randn(batch_size, src_len, d_model).cuda()
    mask = torch.randn(batch_size, tgt_len, src_len).cuda()
    out = model(q, k, v, mask)
    print(out)