# -*- coding: UTF-8 -*-
# seq2seq.py
# @Time     : 15/May/2019
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from emg.models.torch_model import start_train


hyperparameters = {
    'input_size': (128,),
    'hidden_size': 256,
    'seq_length': 10,
    'seq_result': False,
    'frame_input': False
}


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    # value = torch.Tensor(sinusoid_table)
    value = torch.from_numpy(sinusoid_table)
    return value.to('cuda')


def get_attn_key_pad_mask(seq_q):
    """ For masking out the padding part of key sequence. """
    padding_mask = torch.zeros((seq_q.size(0), seq_q.size(1), seq_q.size(1)), dtype=torch.uint8)
    return padding_mask


def get_non_pad_mask(seq):
    value = torch.ones((seq.size(0), seq.size(1), 1), requires_grad=True)
    return value
    # assert seq.dim() == 2
    # return seq.ne(0).type(torch.float).unsqueeze(-1)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(self,
                 len_seq, d_word_vec,
                 n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1):
        super().__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(len_seq+1, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        src_seq = src_seq.to('cuda')  # BUG: 不能在cuda上训练
        slf_attn_mask = get_attn_key_pad_mask(src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        pos_emb = self.position_enc(src_pos)
        pos_emb = pos_emb.to(torch.float32)
        enc_output = src_seq + pos_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output


class Decoder(nn.Module):
    def __init__(self, seq_len, gesture_num):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(seq_len*128, seq_len*128)
        self.output = nn.Linear(seq_len*128, gesture_num)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.output(x)


class Transformer(nn.Module):
    def __init__(self, seq_len,
                 d_word_vec=128, d_model=128, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()
        self.encoder = Encoder(
            len_seq=seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(seq_len, 8)

    def forward(self, src_seq):
        src_pos = torch.arange(1, src_seq.size(1)+1)
        src_pos = src_pos.repeat(src_seq.size(0))
        src_pos = src_pos.view(-1, src_seq.size(1))
        src_pos = src_pos.to('cuda')
        enc_output = self.encoder(src_seq, src_pos)
        # print(src_seq.size())
        # print(enc_output.size())
        dec_output = self.decoder(enc_output.view(-1, src_seq.size(1)*128))
        return dec_output


def main(train_args, test_mode=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = Transformer(args['seq_length'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    start_train(args, model, optimizer, test_mode)


if __name__ == "__main__":
    test_args = {
        'model': 'seq2seq',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 10,
        'train_batch_size': 64,
        'val_batch_size': 1024,
        'stop_patience': 5,
        'log_interval': 100,
        'load_model': False
    }

    main(test_args, test_mode=False)
