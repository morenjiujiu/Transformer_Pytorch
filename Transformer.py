#encoding:utf-8
import tensorflow as tf
import numpy as np
import utils
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    :param
        q: Queries张量，形状为[B, L_q, D_q]
        k: Keys张量，形状为[B, L_k, D_k]
        v: Values张量，形状为[B, L_v, D_v]
        mask: Masking张量，形状为[B, L_q, L_k]
        其中D_q=D_k=D_v, L_q=L_q=L_v
    :return
        attention张量
    """
    def __init__(self,dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,q,k,v,mask):
        dk = k.size()[-1]  # k的dim
        score = torch.bmm(q, k.transpose(1,2)) * np.sqrt(dk)  #这里score 的shape为[B,L_q,L_k]
        if mask is not None:
            mask= mask.repeat(score.size()[0]/mask.size()[0], 1, 1)  #mask和score大小要相同

            score = score.masked_fill_(mask, -np.inf)
            #masked_fill_( mask, value) 在mask的值为1的地方用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同。
            #mask是0-1的ByteTensor

        attention = self.softmax(score)
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention) # replace nan caused softmax of all -inf
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context


class MultiHeadAttention(nn.Module):
    """
    :param
        model_dim 就是k的dim  D_k=D_q=D_v=model_dim
        n_head 是heads的数量
    :return multi_head_attention张量
    """
    def __init__(self, model_dim, n_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.model_dim=model_dim
        self.n_head=n_head
        self.head_dim = self.model_dim // self.n_head
        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head) #size of each input sample, size of each output sample
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head) #这里的输出变成了多层
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_final=nn.Linear(self.head_dim * self.n_head, self.model_dim) #todo
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)


    def forward(self, query, key, value, mask=None):
        q = self.linear_q(query)  # [B, L_q, head_dim*n_head]
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size=k.size()[0]

        q_ = q.view(batch_size * self.n_head, -1, self.head_dim)
        k_ = k.view(batch_size * self.n_head, -1, self.head_dim)
        v_ = v.view(batch_size * self.n_head, -1, self.head_dim)

        context = self.scaled_dot_product_attention(q_, k_, v_, mask)

        o = context.view(batch_size, -1, self.head_dim * self.n_head)
        o = self.linear_final(o)
        o = self.dropout(o)
        return o


'''这个是图中的那个 Feed Forward'''
class PositionWiseFFN(nn.Module):
    def __init__(self,model_dim,dropout_rate):
        super(PositionWiseFFN,self).__init__()
        self.conv1=nn.Conv1d(model_dim, 126,kernel_size=1)
        self.conv2=nn.Conv1d(126, model_dim,kernel_size=1)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x):
        x=x.transpose(1,2) #todo
        o=F.relu(self.conv1(x))
        o=self.conv2(o)
        o=self.dropout(o)
        return o



class Encoder(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate, n_layer):
        super(Encoder,self).__init__()
        self.n_layer = n_layer
        self.lnorm = nn.LayerNorm(model_dim)
        self.multi_head=MultiHeadAttention(model_dim,n_head,dropout_rate)
        self.position_wise_ffn=PositionWiseFFN(model_dim, dropout_rate)

    def forward(self, xz, mask):
        for n in range(self.n_layer):
            xz = self.lnorm(self.multi_head(xz, xz, xz, mask) + xz)
            xz = self.lnorm(self.position_wise_ffn(xz).transpose(1, 2)  + xz)
        return xz



class Decoder(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate, n_layer):
        super(Decoder,self).__init__()
        self.n_layer=n_layer
        self.lnorm = nn.LayerNorm(model_dim)
        self.multi_head = MultiHeadAttention(model_dim, n_head, dropout_rate)
        self.position_wise_ffn = PositionWiseFFN(model_dim,dropout_rate)

    def forward(self, xz, yz, mask):
        for n in range(self.n_layer):
            yz = self.lnorm(self.multi_head(yz, yz, yz, mask) + yz)
            yz = self.lnorm(self.multi_head(yz, xz, xz, None) + yz)
            yz = self.lnorm(self.position_wise_ffn(yz).transpose(1, 2)  + yz)
        return yz


class Transformer(nn.Module):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, dropout_rate, padding_idx=0):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.padding_idx = torch.tensor(padding_idx)
        self.n_vocab=n_vocab

        self.embeddings =torch.nn.Embedding(n_vocab, embedding_dim=model_dim)
        self.build_encoder=Encoder(model_dim, n_head, dropout_rate, n_layer)
        self.build_decoder=Decoder(model_dim, n_head, dropout_rate, n_layer)
        self.linear=nn.Linear(model_dim, n_vocab)
        self.dropout=nn.Dropout(dropout_rate)


    def position_embedding(self):
        pos = np.arange(self.max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(self.model_dim)[None, :] / self.model_dim)  # [max_len, model_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]
        return torch.tensor(pe, dtype=torch.float32)


    def pad_mask(self, seqs):
        mask = torch.where(torch.tensor(torch.equal(seqs, self.padding_idx) ),
                           torch.zeros_like(seqs),
                           torch.ones_like(seqs))  # 0 idx is padding
        return torch.tensor(torch.unsqueeze(mask, dim=1) * torch.unsqueeze(mask, dim=2), dtype=torch.uint8)


    def output_mask(self, seqs):  #sequence mask
        batch_size, seq_len = seqs.size()
        #下三角矩阵，下面为1
        mask = ~torch.triu(torch.ones((self.max_len, self.max_len), dtype=torch.uint8), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        # print "mask",mask
        return mask


    def forward(self, tfx, tfy):
        x_embedded=self.embeddings(tfx)+ self.position_embedding() #[n, step, dim]
        y_embedded=self.embeddings(tfy[:, :-1])+ self.position_embedding() #[n, step, dim]
        # print "x_embedded.size",x_embedded.size(),"y_embedded.size",y_embedded.size()

        encoded_z = self.build_encoder(x_embedded, mask=self.pad_mask(tfx))
        # print "encoded_z.size",encoded_z.size()
        decoded_z = self.build_decoder(y_embedded, encoded_z, mask=self.output_mask(tfy[:, :-1]))

        logits = self.linear(decoded_z)
        return logits




# get and process data
vocab, x, y, v2i, i2v, date_cn, date_en = utils.get_date_data()
print("Chinese time order: ", date_cn[:3], "\nEnglish time order: ", date_en[:3])
print("vocabularies: ", vocab)
print("x index sample: \n", x[:2], "\ny index sample: \n", y[:2])
MODEL_DIM =  32
MAX_LEN = 12
N_LAYER =  3
N_HEAD = 4
DROPOUT_RATE = 0.1

model = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, len(vocab), DROPOUT_RATE)
criterion = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)



#train
import time
t0 = time.time()
for t in range(2000):
    bi = np.random.randint(0, len(x), size=64)
    bx, by = utils.pad_zero(x[bi], max_len=MAX_LEN), utils.pad_zero(y[bi], max_len=MAX_LEN + 1)

    bx_ = torch.tensor(bx, dtype=torch.int64)
    by_ = torch.tensor(by, dtype=torch.int64)

    logits=model(bx_, by_)

    # print "logits.size",logits.size()
    # print "by_[:1, :].size",by_[:1, :].size()
    # print "by_.size",by_.size()
    # print "by_[:, 1:].size",by_[:, 1:].size()

    loss = criterion(logits.transpose(1,2), by_[:, 1:])
    # loss = criterion(logits, by_[:, 1:])

    if t % 50==0:
        logits_=model(bx_[:1, :], by_[:1, :])

        t1 = time.time()
        print(
            "step: ", t,
            "| time: %.2f" % (t1-t0),
            "| loss: %.3f" % loss.item(),
            "| target: ", "".join([i2v[i] for i in by[0, 1:] if i != v2i["<PAD>"]]),
            "| inference: ", "".join([i2v[i] for i in np.argmax(logits_[0].detach().numpy(), axis=1) if i != v2i["<PAD>"]]),
        )
        t0 = t1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



#prediction
with torch.no_grad():
    src_seq = "05-08-30"
    src_pad = utils.pad_zero(np.array([v2i[v] for v in src_seq])[None, :], MAX_LEN)
    tgt_seq = "<GO>"
    tgt = utils.pad_zero(np.array([v2i[tgt_seq], ])[None, :], MAX_LEN + 1)
    tgti = 0

    while True:
        logit = model(torch.tensor(src_pad, dtype=torch.int64), torch.tensor(tgt, dtype=torch.int64))[0, tgti, :]
        idx = np.argmax(logit.detach().numpy())
        tgti += 1
        tgt[0, tgti] = idx
        if idx == v2i["<EOS>"] or tgti >= MAX_LEN:
            break
    pred_seq="".join([i2v[i] for i in tgt[0, 1:tgti]])
    print("src: ", src_seq, "| prediction: ", pred_seq)



