# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import overlap_and_add
from ..utils import capture_init
from AttentionDE.models.transformer import SBTransformerBlock


class MulCatBlock(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=False):
        super(MulCatBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(input_size, hidden_size, 1, dropout=dropout,
                           batch_first=True, bidirectional=bidirectional)
        self.rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)

        self.gate_rnn = nn.LSTM(input_size, hidden_size, num_layers=1,
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.gate_rnn_proj = nn.Linear(
            hidden_size * self.num_direction, input_size)

        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input):
        output = input
        # run rnn module
        rnn_output, _ = self.rnn(output)
        rnn_output = self.rnn_proj(rnn_output.contiguous(
        ).view(-1, rnn_output.shape[2])).view(output.shape).contiguous()
        # run gate rnn module
        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = self.gate_rnn_proj(gate_rnn_output.contiguous(
        ).view(-1, gate_rnn_output.shape[2])).view(output.shape).contiguous()
        # apply gated rnn
        gated_output = torch.mul(rnn_output, gate_rnn_output)
        gated_output = torch.cat([gated_output, output], 2)
        gated_output = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])).view(output.shape)
        return gated_output


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


class DPMulCat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_spk,
                 dropout=0, num_layers=1, bidirectional=True, input_normalize=False):
        super(DPMulCat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        self.intra_numlayers = 4
        self.inter_numlayers = 4
        self.encoder_out_nchannels = 128
        self.intra_nhead = 8
        self.inter_nhead = 8
        self.intra_dffn = 512
        self.inter_dffn = 512
        self.intra_use_positional = True
        self.inter_use_positional = True
        self.intra_norm_before = True
        self.inter_norm_before = True

        self.intra_model = SBTransformerBlock(
            num_layers=self.intra_numlayers,
            d_model=self.encoder_out_nchannels,
            nhead=self.intra_nhead,
            d_ffn=self.intra_dffn,
            use_positional_encoding=self.intra_use_positional,
            norm_before=self.intra_norm_before,
        )

        self.inter_model = SBTransformerBlock(
            num_layers=self.inter_numlayers,
            d_model=self.encoder_out_nchannels,
            nhead=self.inter_nhead,
            d_ffn=self.inter_dffn,
            use_positional_encoding=self.inter_use_positional,
            norm_before=self.inter_norm_before,
        )

        # create the dual path pipeline
        for i in range(num_layers):
            # self.rows_grnn.append(MulCatBlock(
            #     input_size, hidden_size, dropout, bidirectional=bidirectional))
            # self.cols_grnn.append(MulCatBlock(
            #     input_size, hidden_size, dropout, bidirectional=bidirectional))

            self.rows_grnn.append(self.intra_model)
            self.cols_grnn.append(self.inter_model)

            if self.in_norm:
                self.rows_normalization.append(
                    nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(
                    nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1))

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(
                batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(
                batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
            row_output = self.rows_normalization[i](row_output)
            # apply a skip connection
            if self.training:
                output = output + row_output
            else:
                output += row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(
                batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(
                batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
            col_output = self.cols_normalization[i](col_output).contiguous()
            # apply a skip connection
            if self.training:
                output = output + col_output
            else:
                output += col_output

            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)
        return output_all


class Separator(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2,
                 layer=4, segment_size=100, input_normalize=False, bidirectional=True):
        super(Separator, self).__init__()

        # input_dim = 385
        # feature_dim = 128
        # hidden_dim = 128
        # output_dim = 257
        # num_spk = 2
        # layer = 6
        # segment_size = 126

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(self.feature_dim, self.hidden_dim,
                                  self.feature_dim, self.num_spk, num_layers=layer, bidirectional=bidirectional,
                                  input_normalize=input_normalize)

    # ======================================= #
    # The following code block was borrowed and modified from https://github.com/yluo42/TAC
    # ================ BEGIN ================ #
    def pad_segment(self, input, segment_size):
        # [1, 128, 7999], 126
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len %
                               segment_size) % segment_size

        # rest = 2
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)
                           ).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, dim, segment_stride)).type(input.type())

        # pad_aux.shape = [1, 128, 63]
        input = torch.cat([pad_aux, input, pad_aux], 2)
        # input.shape = [1, 128, 8127])
        return input, rest

    def create_chuncks(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size,
                                                                    dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(
            batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(
            batch_size, dim, -1, segment_size).transpose(2, 3)
        return segments.contiguous(), rest

    def merge_chuncks(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(
            batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(
            batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(
            batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()  # B, N, T

    # ================= END ================= #

    def forward(self, input):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(
            input, self.segment_size)

        # enc_segments.shape = [1, 128, 126, 128]
        # enc_rest = 2

        # separate
        output_all = self.rnn_model(enc_segments)

        # merge back audio files
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(
                output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


class SWave(nn.Module):
    @capture_init
    def __init__(self, N, L, H, R, C, sr, segment, input_normalize):
        super(SWave, self).__init__()
        # hyper-parameter
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = N, L, H, R, C, sr, segment
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        # similar to dprnn paper, setting chancksize to sqrt(2*L)
        self.segment_size = int(
            np.sqrt(2 * self.sr * self.segment / (self.L / 2)))

        # model sub-networks
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(self.filter_dim + self.N, self.N, self.H,
                                   self.filter_dim, self.num_spk, self.layer, self.segment_size, self.input_normalize)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)

        # fix time dimension, might change due to convolution operations
        T_mix = mixture.size(-1)
        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(
                mixture.shape[0], self.C, self.N, mixture_w.shape[2])
            output_ii = self.decoder(output_ii)

            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            outputs.append(output_ii)
        return torch.stack(outputs)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


import numpy as np


class STrans(nn.Module):
    def __init__(self, in_channels):
        super(STrans, self).__init__()
        self.in_channels = in_channels
        self.Wqs = nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.Wks = nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.Wvs = nn.Parameter(torch.Tensor(in_channels, in_channels))

        torch.nn.init.xavier_uniform_(self.Wqs)
        torch.nn.init.xavier_uniform_(self.Wks)
        torch.nn.init.xavier_uniform_(self.Wvs)

    def forward(self, xs):
        q_h = torch.matmul(xs, self.Wqs)
        k_h = torch.matmul(xs, self.Wks)
        v_h = torch.matmul(xs, self.Wvs)

        Q_h = q_h.view(1, -1, 8, 16)
        K_h = k_h.view(1, -1, 8, 16)
        V_h = v_h.view(1, -1, 8, 16)

        Q_K_h = Q_h * K_h
        Q_K_h_e = F.softmax(Q_K_h / np.sqrt(self.in_channels), dim=-1)

        return (V_h * Q_K_h_e).view((1, -1, self.in_channels)).permute(0, 2, 1)


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        # setting 50% overlap
        self.conv = nn.Conv1d(
            1, N, kernel_size=L, stride=L // 2, bias=False)

        # DE
        self.conv = nn.Sequential(
            Conv1D(1, N, L, stride=L // 2),
            nn.ReLU(),
            Conv1D(N, N, kernel_size=5, stride=1, padding=1),
            nn.PReLU(),
            Conv1D(N, N, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            Conv1D(N, N, kernel_size=1, stride=1, padding=1),
            nn.PReLU()
        )

        # Attention
        self.strans = STrans(N)

    def forward(self, mixture):
        # mixture.shape = [1, 32000]
        mixture = torch.unsqueeze(mixture, 1)
        # mixture.shape = [1, 1, 32000]
        #         mixture_w = F.relu(self.conv(mixture))
        # mixture.shape = [1, 128, 7999]

        x = self.conv(mixture)
        x = x.permute(0, 2, 1)
        mixture_w = self.strans(x)
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.L))(est_source)
        est_source = overlap_and_add(est_source, self.L // 2)

        return est_source
