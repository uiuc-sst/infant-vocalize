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
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools

"""
python feature_selection.py --NUM_FEATURES=23 \
--FORWARD=TRUE --FLOATING=TRUE --SINGLE=FALSE \
--TF_RECORDS_DIR='reliability_CHN_segments_tfrecords/train_23.tfrecord'
--CHN_DATASET_DIR='reliability_CHN_segments/'
"""

flags = tf.app.flags
flags.DEFINE_string(
    'TF_RECORDS_DIR', None,
    'tfrecord')
flags.DEFINE_string(
    'CHN_DATASET_DIR', None,
    'tfrecord')
flags.DEFINE_integer(
    'NUM_FEATURES', None,
    'How many features in tfrecord')
FLAGS = flags.FLAGS

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

# load data
data_path = FLAGS.TF_RECORDS_DIR #'/Users/yijiaxu/Desktop/prosody_AED/features_to_be_selected/train.tfrecord'
balanced_train_path = FLAGS.CHN_DATASET_DIR
train_data_num = len(os.listdir(balanced_train_path.split('/')[0]))
batch_size = train_data_num-1
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
# pdb.set_trace()

# for reliability balance:
newx = np.zeros((60*5,23))
newx[:60,:] = img[lbl == 4][:60]
newx[60:120,:] = img[lbl == 3][:60]
newx[120:180,:] = img[lbl == 2][:60]
newx[180:240,:] = img[lbl == 1][:60]
newx[240:300,:] = img[lbl == 0][:60]

newy = np.zeros(60*5)
newy[:60] = 4
newy[60:120] = 3
newy[120:180] = 2
newy[180:240] = 1
newy[240:300] = 0

# 5-way
X = newx #img
y = newy #lbl

# 4-way
# X = X[y!=4]
# y = y[y!=4]

# # 3-way
# reliability
# X = np.concatenate((X[:150,:],X[180:210,:]),axis=0)
# y = np.concatenate((y[:150],y[180:210]),axis=0)

# y[y==1] = 0

# non-reliability
# X = np.concatenate((X[y==0][:int(len(X[y==0])/2)],X[y!=0]),axis=0)
# y = np.concatenate((y[y==0][:int(len(y[y==0])/2)],y[y!=0]),axis=0)

lr = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
y_pred_list = []
y_test_list = []
average_accuracy = 0
average_Fscore = 0
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train,y_train)
    # print sum(lr.predict(X_test) == y_test)*1.0/len(y_test)
    y_pred = lr.predict(X_test)
    average_accuracy += sum(y_pred == y_test)*1.0/len(y_test)
    # print f1_score(y_test, y_pred,average='macro')
    average_Fscore += f1_score(y_test, y_pred, average='macro')
    y_pred_list.extend(y_pred)
    y_test_list.extend(y_test)
average_accuracy /= 5
average_Fscore /= 5
print "average_accuracy:",average_accuracy
print "average_Fscore:",average_Fscore

cnf_matrix = confusion_matrix(y_test_list, y_pred_list)
class_names = ['CRY','FUS','LAU','BAB'] # Why omit HIC?

np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix of 5-way LDA on tokens coded by two labelers')
plt.show()
pdb.set_trace()
np.save('results_npy/cnf_matrix_23_reliability_5way',cnf_matrix)
