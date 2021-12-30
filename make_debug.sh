#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Yossi Adi (adiyoss)

path=egs/LibriMix/tr
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav8k/min/train-360/mix_clean > $path/2spk-C/8kmin/mix_clean.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav8k/min/train-360/mix_both > $path/2spk-C/8kmin/mix_both.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav16k/max/train-360/mix_clean > $path/2spk-C/16kmax/mix_clean.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav16k/max/train-360/mix_both > $path/2spk-C/16kmax/mix_both.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav8k/min/train-360/s1 > $path/2spk-C/8kmin/s1.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav8k/min/train-360/s2 > $path/2spk-C/8kmin/s2.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav16k/max/train-360/s1 > $path/2spk-C/16kmax/s1.json
python -m svoice.data.audio E:/LibriMix-master/git-projectLibriMix-master/Libri2Mix/wav16k/max/train-360/s2 > $path/2spk-C/16kmax/s2.json