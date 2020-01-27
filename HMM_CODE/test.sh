#!/bin/bash

# A subset of run.sh, for debugging.

k=4
i=0.5
j=3

python caveneuwirth.py --TestTextgrid='e20170719_090917_011543_GRP.TextGrid' \
--Train_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/train_CHN_Textgrid/' \
--A_rand_init=False --CsvDirectory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/multi_prob_stats/'$k'label'$j'.csv' \
--lamb=$i --B_norm=False --getBmatrix=multi \
--train=False \
--Test_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/test_CHN_Textgrid/' \
--num_states=$k >> 'e20170719_090917_011543_GRP-'$k'labels'
