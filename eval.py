"""
Usage:
python eval.py \
--filenames=balanced_tfrecords_21/test.tfrecord \
--batch_size=1 --num_epochs=1 \
--ckptdir='ckpt_21/' --restore=True --num_classes=4 \
--CHN_test_files='balanced_CHN_segments_test/' \
--prosody_or_fbank='fbank'
"""

# changed map, logits csv store, accuracy calculations

import os
from tensorflow import logging
import tensorflow as tf
import read_data
import pdb
import CNNmodel
import nets
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools

flags = tf.app.flags
flags.DEFINE_string(
    'filenames', None,
    'Path to *tfrecord') # 'tfrecords/output001.wav.tfrecord'
# each example has 100 frames -> 10x100 frame examples
flags.DEFINE_integer("batch_size", 1, "How many examples to process per batch for training.")
flags.DEFINE_integer("num_epochs", 1, "How many examples to process per batch for training.")
flags.DEFINE_string(
    'ckptdir', None,
    'Path to ckpt') # 'ckpt/'
flags.DEFINE_bool(
    'restore', True,
    'If you want to restore the ckpt')
flags.DEFINE_integer(
    'num_classes', 4,
    'The number of classes')
flags.DEFINE_string(
    'CHN_test_files', None,
    'Path to CHN_test_files')
flags.DEFINE_string(
    'prosody_or_fbank', None,
    'Choose the feature type')
FLAGS = flags.FLAGS

def map(string_label):
    if string_label==0: return 'CRY'
    if string_label==1: return 'FUS'
    if string_label==2: return 'LAU'
    if string_label==3: return 'BAB'
    if string_label==4: return 'HIC'
    else: return

# # number of cases, number of correctly classified
# miss = {
# 'CRY':{'CRY':0,'FUS':0,'LAU':0,'BAB':0},
# 'FUS':{'CRY':0,'FUS':0,'LAU':0,'BAB':0},
# 'LAU':{'CRY':0,'FUS':0,'LAU':0,'BAB':0},
# 'BAB':{'CRY':0,'FUS':0,'LAU':0,'BAB':0}}
def plot_confusion_matrix(
    cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(_):
    # Log the version.
    logging.set_verbosity(tf.logging.INFO)
    if FLAGS.filenames:
        filenames= FLAGS.filenames
    if FLAGS.batch_size:
        batch_size= FLAGS.batch_size
    if FLAGS.num_epochs:
        num_epochs= FLAGS.num_epochs
    # Load data placeholders of [batch_size, 100, 64]
    images, labels, filename = read_data.loadembeddingtest(filenames, batch_size=batch_size, num_epochs=num_epochs, feature_type=FLAGS.prosody_or_fbank)
    # Number of test data
    # CHN_test_files = FLAGS.CHN_test_files
    #1-fold test0: ~500
    test_data_num = sum(1 for _ in tf.python_io.tf_record_iterator(filenames))
    #500 #12768-1326 #len(os.listdir(CHN_test_files.split('/')[0]))
    correct=0
    # build model graph
    model = CNNmodel.CNNmodel(inputs=images,labels=labels,num_classes=FLAGS.num_classes,is_train=False, feature_type=FLAGS.prosody_or_fbank)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.ckptdir)
        saver.restore(sess, latest_checkpoint)
        sess.run([tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        lab_array = []
        pred_array = []
        with open('5label4.csv', 'wb') as csvfile:
            csvfile.write('total_filename,filename,start_time,end_time,true_label,prediction,logits0,logits1,logits2,logits3,logits4')
            csvfile.write('\n')
            for i in xrange(test_data_num): # for each batch
                img, lab, filena, pred, logits = sess.run([images, labels, filename, model._predictions, model._logits])
                # pdb.set_trace()
                # miss[map(lab)][map(pred)]+=1
                start_time = filena.split('-')[-3]
                end_time = filena.split('-')[-2]
                name = filena.split('-')[0]

                # pdb.set_trace()
                # for HMM observation sequence usage:
                csvfile.write(filena+','+name+','+start_time+','+end_time+','+map(lab)+','+map(pred)+','+str(logits[0][0])+','+str(logits[0][1])+','+str(logits[0][2])+','+str(logits[0][3])+','+str(logits[0][4]))
                csvfile.write('\n')
                # correct+=int(lab==pred)
                lab_array.append(lab)
                pred_array.append(pred[0])
            coord.request_stop()
            coord.join(threads)
        lab_array=np.array(lab_array)
        pred_array=np.array(pred_array)
        Fscore = f1_score(lab_array, pred_array,average='macro')
        Accuracy = sum(lab_array==pred_array)*1.0/test_data_num
        print('F-score:',Fscore)
        print('Accuracy:',Accuracy)
        # cnf_matrix = confusion_matrix(lab_array, pred_array)
        # class_names = ['CRY','FUS','LAU','BAB']
        # np.set_printoptions(precision=2)
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Confusion matrix of 4-way CNN on tokens coded by two labelers')
        # plt.show()

if __name__ == '__main__':
    tf.app.run()
