
from multiprocessing import Pool
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import pdb 
import wav_to_spectro
import os
import glob
import params
from tensorflow import gfile 
import tgt
from pydub import AudioSegment
from sklearn.model_selection import KFold
"""
python make_tfrecords_FBANK.py \
--CHN_segment_dir='/Users/yijiaxu/Desktop/prosody_AED/balanced_CHN_segments_train/' \
--tfrecord_file_dir='fbank_tfrecords/' --train=True
"""

flags = tf.app.flags

flags.DEFINE_string(
    'tfrecord_file_dir', None,
    'Path to a TFRecord file dir.')
flags.DEFINE_string(
    'CHN_segment_dir', None,
    'Path to a extracted CHN segments file dir.')
flags.DEFINE_bool(
    'train',None,
    'if it is train or test.')

FLAGS = flags.FLAGS

def map(string_label):
  if string_label=='CRY': return 0
  if string_label=='FUS': return 1
  if string_label=='LAU': return 2
  if string_label=='BAB': return 3
  if string_label=='HIC': return 4
  else: return  

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if not os.path.exists(FLAGS.tfrecord_file_dir):
  os.makedirs(FLAGS.tfrecord_file_dir)

if FLAGS.train:
  files = np.array(glob.glob(FLAGS.CHN_segment_dir+"*.wav"))
  kf = KFold(n_splits=5, shuffle=True)
  count = 0 
  for train_index, test_index in kf.split(files):
    train_files, test_files = files[train_index], files[test_index]

    #train
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file_dir+'train'+str(count)+'.tfrecord')
    for file in train_files:
      filename_list = (file.split('.wav')[0]).split('-')

      # minor modifications
      # if filename_list[3]=='HIC':
      #   continue 
      examples_batch = wav_to_spectro.wavfile_to_examples(file)
      # pdb.set_trace()
      if examples_batch.shape[0]!=0:
        label_batch = []
        label_batch.append(map(filename_list[3]))
        labels = np.array(label_batch)
        for image_idx,image in enumerate(examples_batch):
          image=image.astype("float32") 
          feature = {'data': _bytes_feature((image.tostring())),'label': _bytes_feature((labels.tostring())),'filename':_bytes_feature(file.split('/')[-1].split('.wav')[0])}
          example = tf.train.Example(features=tf.train.Features(feature=feature)) 
          writer.write(example.SerializeToString())
    writer.close() 

    #test
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file_dir+'test'+str(count)+'.tfrecord')
    for file in test_files:
      examples_batch = wav_to_spectro.wavfile_to_examples(file)
      filename_list = (file.split('.wav')[0]).split('-')
      # minor modifications
      # if filename_list[3]=='HIC':
      #   continue 
      if examples_batch.shape[0]!=0:
        label_batch = []
        label_batch.append(map(filename_list[3]))
        labels = np.array(label_batch)
        examples_batch=examples_batch.astype("float32")  
        feature = {'data': _bytes_feature((examples_batch.tostring())),'label': _bytes_feature((labels.tostring())),'filename':_bytes_feature(file.split('/')[-1].split('.wav')[0])}
        example = tf.train.Example(features=tf.train.Features(feature=feature)) 
        writer.write(example.SerializeToString())
    writer.close() 

    count+=1

if not FLAGS.train: # in this typical case, generate all the files for HMM probability use
  files = np.array(glob.glob(FLAGS.CHN_segment_dir+"*.wav"))

  writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file_dir+'test'+'.tfrecord')
  for file in files:
    examples_batch = wav_to_spectro.wavfile_to_examples(file)
    filename_list = (file.split('.wav')[0]).split('-')
    # minor modifications
    if filename_list[3]=='HIC':
      continue 
    if examples_batch.shape[0]!=0:
      label_batch = []
      label_batch.append(map(filename_list[3]))
      labels = np.array(label_batch)
      examples_batch=examples_batch.astype("float32")  
      feature = {'data': _bytes_feature((examples_batch.tostring())),'label': _bytes_feature((labels.tostring())),'filename':_bytes_feature(file.split('/')[-1].split('.wav')[0])}
      example = tf.train.Example(features=tf.train.Features(feature=feature)) 
      writer.write(example.SerializeToString())
  writer.close()     


  # writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file_dir+'test'+'.tfrecord')
  # for file in glob.glob(FLAGS.CHN_segment_dir+"*.wav"):
  #   examples_batch = wav_to_spectro.wavfile_to_examples(file)
  #   if examples_batch.shape[0]!=0:
  #     filename_list = (file.split('.wav')[0]).split('-')
  #     label_batch = []
  #     label_batch.append(map(filename_list[3]))
  #     labels = np.array(label_batch)
  #     examples_batch=examples_batch.astype("float32")  
  #     feature = {'data': _bytes_feature((examples_batch.tostring())),'label': _bytes_feature((labels.tostring())),'filename':_bytes_feature(file.split('/')[-1].split('.wav')[0])}
  #     example = tf.train.Example(features=tf.train.Features(feature=feature)) 
  #     writer.write(example.SerializeToString())
  # writer.close() 





