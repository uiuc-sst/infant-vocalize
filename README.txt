Total balanced files from wav dataset, and corresponding tfrecords
total_balanced_CHN, and balanced_tfrecords_23/train.tfrecord

LDA classifier results on total balanced segments from one annotator:
python LDA_test_reliability.py  --TF_RECORDS_DIR='balanced_tfrecords_23/train.tfrecord' --CHN_DATASET_DIR='total_balanced_CHN/' --NUM_FEATURES=23

LDA classifier results on balanced reliability dataset:
LDA_test_reliability.py --NUM_FEATURES=23  --TF_RECORDS_DIR='reliability_CHN_segments_tfrecords/train_23.tfrecord'  --CHN_DATASET_DIR='reliability_CHN_segments/'

Make FBANK features:
python make_tfrecords_FBANK.py \
--CHN_segment_dir='/Users/yijiaxu/Desktop/prosody_AED/total_balanced_CHN/' \
--tfrecord_file_dir='fbank_tfrecords_5labels/' --train=True

(fbank_tfrecords_5labels: now we get 5 set of train-test pair for performing CV evaluations, on 5 labels)
(fbank_tfrecords_4labels: ignored HIC by minor modifications on code)

(50x64)
// (fban_features_4labels_smaller): smaller FBANK features (20x64)

Train with FBANK features:
    python train_prosody.py \
    --filenames= fbank_tfrecords_5labels/train0.tfrecord \
    --batch_size=30 --num_epochs=1200 \
    --logDir='ckpt_fbank/' --restore=False --num_classes=5
    --prosody_or_fbank=’fbank’

Evaluate the features (accuracy and F-score):
    python eval.py --filenames=fbank_tfrecords_5labels/test0.tfrecord --batch_size=1 --num_epochs=1 --ckptdir='ckpt_fbank/ 4labels0' --restore=True --num_classes=5 --prosody_or_fbank='fbank'

Make 5-fold cv 23 prosody features for NN classifier:
    Make_tfrecord.py -> prosody_tfrecords_4labels

RESULTS:
TRAIN CNN:
    python train_prosody.py --filenames=fbank_tfrecords_4labels_smaller/train0.tfrecord --batch_size=16 --num_epochs=256 --logDir='ckpt_fbank/' --restore=False --num_classes=4 --prosody_or_fbank='fbank'

TEST CNN(K-FOLD):
    python eval.py --filenames=fbank_tfrecords_4labels_smaller/test0.tfrecord --batch_size=1 --num_epochs=1 --ckptdir='ckpt_fbank/' --restore=True --num_classes=4 --prosody_or_fbank='fbank'

Use the CNN model trained to HMM:
First get all (all segments in all file sequences) the observation probabilities by using 5-fold different CNN models (in fbank_tfrecords_total_4labels/test.tfrecord)
    python make_tfrecords_FBANK.py --CHN_segment_dir='/Users/yijiaxu/Desktop/prosody_AED/CHN_segments/' --tfrecord_file_dir='fbank_tfrecords_total_4labels/' --train=False

Run eval on it. -> prob_hmm.csv
NOTE: run on all segments (CHN_TOTAL) not the balanced one
    python eval.py --filenames=fbank_tfrecords_total_4labels/test.tfrecord --batch_size=1 --num_epochs=1 --ckptdir='ckpt_fbank/' --restore=True --num_classes=4 --prosody_or_fbank='fbank'

    python caveneuwirth.py --TestTextgrid='e20170801_122506_011543_GRP.TextGrid' --Train_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/test_CHN_Textgrid/' --A_rand_init=False --CsvDirectory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/multi_prob_stats/' --lamb=0 --B_norm=False --getBmatrix=multi --train=False --Test_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/test_CHN_Textgrid/'

# python /Users/yijiaxu/Library/Python/2.7/bin/tensorboard --logdir=ckpt/
# http://localhost:6006/#scalars
