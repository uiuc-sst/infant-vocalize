#!/bin/bash

for k in 4 5
do
	for i in 0 0.3 0.5 1.0 1.2
	do
		for j in 0 1 2 3 4 
		do
			python caveneuwirth.py --TestTextgrid='e20170719_090917_011543_GRP.TextGrid' \
			--Train_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/train_CHN_Textgrid/' \
			--A_rand_init=False --CsvDirectory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/multi_prob_stats/'$k'label'$j'.csv' \
			--lamb=$i --B_norm=False --getBmatrix=multi \
			--train=False \
			--Test_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/test_CHN_Textgrid/' \
			--num_states=$k >> 'e20170719_090917_011543_GRP-'$k'labels'
		done
	done
done

# TO DO: run each test file, remember to change the correpsonding training files 