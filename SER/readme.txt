Speech emotion recognition trained on RAVDESS dataset.
model_py/cnn_multi_filters.py --architecture of CNN and attention
model/RAVDESS/emo_cnn_multi_att.pt --pretrained weights
ser_ravdess.py --entry point of the program
util.py --necessary helper functions

The state-of-the-art results for ravdess dataset is trained on 40-dimensional filterbanks for every 25ms audio with 10ms shift.
