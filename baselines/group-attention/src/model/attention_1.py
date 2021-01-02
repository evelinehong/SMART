import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from test import group_mask,src_to_mask


class Attention_1(nn.Module):
    def __init__(self, dim):
        super(Attention_1, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        '''
        output: decoder,  (batch, 1, hiddem_dim2)
        context: from encoder, (batch, n, hidden_dim1)
        actually, dim2 == dim1, otherwise cannot do matrix multiplication
        bi-linear
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (b, o, dim) * (b, dim, i) -> (b, o, i)
        attn = torch.bmm(output, context.transpose(1,2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # (b, o, i) * (b, i, dim) -> (b, o, dim)
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size)))\
                            .view(batch_size, -1, hidden_size)
        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn


class Attention_2(nn.Module):
    def __init__(self, dim):
        '''
        You have to know, actually, due to my laziness, I named the basic self attention as attention_2
        :param dim:
        '''
        super(Attention_2, self).__init__()
        self.linear_out = nn.Linear(2*dim, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        '''
        output: decoder,  (batch, 1, hiddem_dim2)
        context: from encoder, (batch, n, hidden_dim1)
        actually, dim2 == dim1, otherwise cannot do matrix multiplication
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (b, o, dim) * (b, dim, i) -> (b, o, i)
        attn = torch.bmm(output, context.transpose(1,2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (b, o, i) * (b, i, dim) -> (b, o, dim)
        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)

        #print ("combine:", combined.size(),  'hidden_size:', hidden_size, 'output:',output.size())
        #combine: torch.Size([256, 71, 2048]) hidden_size: 1024 output: torch.Size([256, 71, 1024])
        output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size)))\
                            .view(batch_size, -1, hidden_size)

        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self,src,pad=0):
        mask = src_to_mask(src)
        self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
        self.final = torch.cat((self.src_mask_between.cuda(),self.src_mask_self.cuda(),self.src_mask_global.to(torch.uint8).cuda(),self.src_mask_question.cuda()),1)
        return self.final.cuda()

    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)
        "Implements Figure 2"

        if mask is not None and len(mask.shape)<4:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # which is linears(query, key, value)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
