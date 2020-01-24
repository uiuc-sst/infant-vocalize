import numpy as np
import os
import tensorflow as tf
import pdb
import random
import shutil

"""
python make_balance_datasets.py \
--CHN_segments_diretory='/Users/yijiaxu/Desktop/prosody_AED/CHN_segments/' \
--Create_balanced_train=True \
--Dest_directory='balanced_CHN_segments_train/'
"""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'CHN_segments_diretory', None,
    'Path to a CHN_segments_diretory')
flags.DEFINE_bool(
    'Create_balanced_train', True,
    'Create train or test sets')
flags.DEFINE_string(
    'Dest_directory', None,
    'Path to created directory')

indir = FLAGS.CHN_segments_diretory
train = FLAGS.Create_balanced_train
dest = FLAGS.Dest_directory

# test_file_name='e20170719_090917_011543_GRP'
m = {'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}
fm = {
    'e20170927_104958_011543_MB':{'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}, \
    'e20170801_122506_011543_GRP':{'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}, \
    'e20170823_154650_011542_RD':{'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}, \
    'e20170906_165204_011543_LC':{'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}, \
    'e20170719_090917_011543_GRP':{'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}
    }

random_files = []
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if f == '.DS_Store':
            continue
        # if train:
        #     if f.split('-')[0] == test_file_name:
        #         continue
        # if not train:
        #     if f.split('-')[0] != test_file_name:
        #         continue
        text = (f.split('.wav')[0]).split('-')[3]
        if text in m:
            m[text] += 1
            random_files.append(f)
max_num_samples = max(m.values())
min_num_samples = min(m.values())
print m
print "max_samples:",max_num_samples
print "min_samples:",min_num_samples
pdb.set_trace()

# create balanced datasets
if not os.path.exists(dest):
    os.makedirs(dest)
copy_num = 0
num = {'BAB':0,'CRY':0,'FUS':0,'LAU':0,'HIC':0}
random.shuffle(random_files)
for f in random_files:
    text = (f.split('.wav')[0]).split('-')[3]
    if num[text] < min_num_samples:
        shutil.copy(indir+f, dest)
        copy_num += 1
        num[text] += 1
print(num)
print(copy_num)
