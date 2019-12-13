from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pdb
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import RFECV
import math
import plot_histogram

"""
python feature_selection.py --NUM_FEATURES=65 \
--FORWARD=TRUE --FLOATING=TRUE --SINGLE=FALSE
"""

flags = tf.app.flags
flags.DEFINE_integer(
    'NUM_FEATURES', None,
    'How many features in tfrecord')
flags.DEFINE_bool(
    'FORWARD', None,
    'How many features in tfrecord')
flags.DEFINE_bool(
    'FLOATING', None,
    'How many features in tfrecord')
flags.DEFINE_bool(
    'SINGLE', None,
    'if look at only one config')
flags.DEFINE_integer(
    'TF_RECORDS_DIR', None,
    'tfrecord')
FLAGS = flags.FLAGS

# define list of strings map to each feature
input_sample = "prev_event_ID,time_duration,max_pitch,average_pitch,pitch_slope,pitch_offset,inter_quartile_pitch,ZCR_log_pitch,average_loudness,\
            max_loudness,min_loudness,inter_quartile_loudness,prob_voiced,\
            MFCC_mean[0],MFCC_mean[1],MFCC_mean[2],MFCC_mean[3],MFCC_mean[4],MFCC_mean[5],MFCC_mean[6],MFCC_mean[7],\
            MFCC_mean[8],MFCC_mean[9],MFCC_mean[10],MFCC_mean[11],\
            MFCC_min[0],MFCC_min[1],MFCC_min[2],MFCC_min[3],MFCC_min[4],MFCC_min[5],MFCC_min[6],MFCC_min[7],\
            MFCC_min[8],MFCC_min[9],MFCC_min[10],MFCC_min[11],\
            MFCC_max[0],MFCC_max[1],MFCC_max[2],MFCC_max[3],MFCC_max[4],MFCC_max[5],MFCC_max[6],MFCC_max[7],\
            MFCC_max[8],MFCC_max[9],MFCC_max[10],MFCC_max[11],\
            MFCC_inter_quartile[0],MFCC_inter_quartile[1],MFCC_inter_quartile[2],MFCC_inter_quartile[3],MFCC_inter_quartile[4],MFCC_inter_quartile[5],MFCC_inter_quartile[6],MFCC_inter_quartile[7],\
            MFCC_inter_quartile[8],MFCC_inter_quartile[9],MFCC_inter_quartile[10],MFCC_inter_quartile[11],\
            ZCR_mean,ZCR_inter_quartile,ZCR_min,ZCR_max"
feature_names = input_sample.replace(" ","").split(',')

# load data
data_path = '/Users/yijiaxu/Desktop/prosody_AED/features_to_be_selected/train.tfrecord'
balanced_train_path = 'balanced_CHN_segments_train/'
train_data_num = len(os.listdir(balanced_train_path.split('/')[0]))

batch_size = train_data_num
with tf.Session() as sess:
	filename_queue = tf.train.string_input_producer(
        [data_path], num_epochs=1)
	reader = tf.TFRecordReader()
  	_, serialized_example = reader.read(filename_queue)
  	features = tf.parse_single_example(
      	serialized_example,
      	features={
        	'data': tf.FixedLenFeature([], tf.string),
        	'label': tf.FixedLenFeature([], tf.string),
        	'filename': tf.FixedLenFeature([], tf.string)
        	},
        )
	image = tf.decode_raw(features['data'], tf.float32)
	image = tf.reshape(image, [FLAGS.NUM_FEATURES])
	label = tf.decode_raw(features['label'], tf.int64)
  	label = label[0]
	filename = features['filename']
	images, labels, filenames = tf.train.shuffle_batch([image, label, filename], batch_size=batch_size, capacity=30, num_threads=1, min_after_dequeue=10)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	img, lbl = sess.run([images, labels])

X = img
y = lbl
# PLOT HISTOGRAMS:
# for i in range(0,65,1):
# 	plot_histogram.plot_hist(X,y,feature_names,i,i+4)

pdb.set_trace()
# lr = LinearRegression()
# lr = LogisticRegression()
# lr = svm.SVC(kernel='linear')
lr =  LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001) #(solver='lsqr',shrinkage='auto')
# lr = svm.LinearSVC(multi_class='crammer_singer',random_state=12345)
# lr = QuadraticDiscriminantAnalysis()

if FLAGS.SINGLE:
	sfs = SFS(lr,
	           k_features=(1,64), #(1,64) # SFS will consider return any feature combination between min and max that scored highest in cross-validtion
	           forward=FLAGS.FORWARD, # forward or backward
	           floating=FLAGS.FLOATING, # put back?
	           verbose=0,
	           scoring='accuracy', #'neg_mean_squared_error',
	           cv=5)
	sfs = sfs.fit(X, y)

	best_feature_index=sfs.k_feature_idx_
	best_feature_name = [feature_names[i] for i in best_feature_index]
	print "The number of best features is:", len(best_feature_index)
	print "The best features' index are:", best_feature_index
	print "The best features are:", best_feature_name

	fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
	config = {(True,True):('FORWARD','FLOATING'),\
	(True,False):('FORWARD','NFLOATING'),\
	(False,True):('BACKWARD','FLOATING'),\
	(False,False):('BACKWARD','NFLOATING'),}
	fig.savefig('feature_selection/SINGLE-'+config[(FLAGS.FORWARD,FLAGS.FLOATING)][0]+'-'+config[(FLAGS.FORWARD,FLAGS.FLOATING)][1])

else:
	config = {(True,True):('FORWARD','FLOATING'),\
	(True,False):('FORWARD','NFLOATING'),\
	(False,True):('BACKWARD','FLOATING'),\
	(False,False):('BACKWARD','NFLOATING'),}
	
	best_feature_index_array = config.copy()
	classifiers = config.copy()
	for i,j in config:
		sfs = SFS(lr,
		           k_features=(1,64), #(1,64) # SFS will consider return any feature combination between min and max that scored highest in cross-validtion
		           forward=i, # forward or backward
		           floating=j, # put back?
		           verbose=0,
		           scoring='accuracy', #'neg_mean_squared_error',
		           cv=5)
		sfs = sfs.fit(X, y)			
		classifiers[(i,j)] = sfs
		best_feature_index_array[(i,j)] = sfs.k_feature_idx_
	best_overlap_features = reduce(set.intersection, (set(val) for val in best_feature_index_array.values()))
	best_feature_name = [feature_names[i] for i in best_overlap_features]
	print "The number of best features is:", len(best_feature_name)
	print "The best features' index are:", best_overlap_features
	print "The best features are:", best_feature_name
pdb.set_trace()

import pandas as pd
pd.DataFrame.from_dict(sfs.get_metric_dict()).T

sfs.subsets_ # the selected feature indices at each step
sfs.k_feature_idx_ # indices of the 3 best features
sfs.k_score_ # prediction score for these 3 features
pdb.set_trace()

# (0, 1, 4, 5, 7, 8, 11, 13, 14, 16, 25, 26, 29, 36, 39, 47, 59, 60, 63) - forward
# (0, 1, 3, 4, 5, 8, 9, 11, 13, 14, 16, 24, 25, 26, 27, 29, 32, 36, 38, 39, 49, 57, 59, 60, 61, 63) - backward
# TODO: check both forward and backward with different configs, return the matched oness(features)
