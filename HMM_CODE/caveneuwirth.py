import re
import string
import numpy as np
import sys
import logging
import myhmm
from confusionmatrix import plotconf
import pdb
import tgt
import csv
import tensorflow as tf
import hmm_util
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
# import sys

# orig_stdout = sys.stdout
# f = open('out.txt', 'w')
# sys.stdout = f

'''
python caveneuwirth.py \
--TestTextgrid='e20170719_090917_011543_GRP.TextGrid' \
--Train_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/train_CHN_Textgrid/' \
--A_rand_init=False \
--CsvDirectory='/Users/yijiaxu/Desktop/prosody_AED/' \
--lamb=1.0 --B_norm=False --getBmatrix=multi \
--train=False --Test_Directory='/Users/yijiaxu/Desktop/prosody_AED/HMM_CODE/test_CHN_Textgrid/'
'''

# tune: normB, lambda, getBmatrix_multi/binary
flags = tf.app.flags
flags.DEFINE_string(
    'getBmatrix', None,
    'Whether use binary or multi to get the B matrix')
flags.DEFINE_string(
    'TestTextgrid', None,
    'Path to TextGridname')
flags.DEFINE_string(
    'Train_Directory', None,
    'Path to TextGrid_Directory')
flags.DEFINE_string(
    'Test_Directory', None,
    'Path to TextGrid_Directory')
flags.DEFINE_string(
    'CsvDirectory', None,
    'Path to CsvDirectory')
flags.DEFINE_integer(
    'num_states', 4,
    'The number of states')
flags.DEFINE_float(
    'lamb', 1,
    'Lambda value')
flags.DEFINE_bool(
    'A_rand_init', True,
    'If you want to randomly initialize the transition matrix')
flags.DEFINE_bool(
    'B_norm', True,
    'If you want to normalize the observation matrix')
flags.DEFINE_bool(
    'train', True,
    'If you want to train or test')
FLAGS = flags.FLAGS

FORMAT = "[%(asctime)s] : %(filename)s.%(funcName)s():%(lineno)d - %(message)s"
DATEFMT = '%H:%M:%S, %m/%d/%Y'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATEFMT)
logger = logging.getLogger(__name__)

def main():
    sys.setrecursionlimit(1500) # Apparently this is a very bad idea.
    # txt, chars = readandclean("textdata.txt")
    TestTextgrid= FLAGS.TestTextgrid
    TextGrid_Directory= FLAGS.Train_Directory
    N = FLAGS.num_states
    A_rand_init = FLAGS.A_rand_init
    CsvDirectory = FLAGS.CsvDirectory
    lamb = FLAGS.lamb
    B_norm = FLAGS.B_norm
    getBmatrix = FLAGS.getBmatrix
    train = FLAGS.train
    Test_Directory = FLAGS.Test_Directory

    if getBmatrix=="multi":
        B, accuracy_original, y, FSCORE_original = hmm_util.getBmatrix_multi(N, Test_Directory, TestTextgrid, CsvDirectory)
    if getBmatrix=="binary":
        B, accuracy_original, y, FSCORE_original = hmm_util.getBmatrix_binary(N, Test_Directory, TestTextgrid, CsvDirectory)

    # This where we set N - states; M or T - observations
    # N = 4

    M = B.shape[1]

    # random initialization of pi
    pi = np.random.rand(N)
    pi = pi / sum(pi)

    # these are matrices
    if A_rand_init:
        A = np.random.rand(N, N)
    else:
        #plug into specific values of A
        # multi LC
        # A = np.array([[ 0.346,  0.225,  0.131,  0.297],
        # [ 0.111,  0.289,  0.188,  0.412],
        # [ 0.056,  0.182,  0.307,  0.455],
        # [ 0.061,  0.152,  0.16 ,  0.627]])

        #multi RD
        # A = np.array([[ 0.483,  0.211,  0.153,  0.152],
        # [ 0.13 ,  0.271,  0.277,  0.322],
        # [ 0.104,  0.169,  0.37 ,  0.356],
        # [ 0.068,  0.158,  0.174,  0.599]])

        #binary RD - norm
        # A = np.array([[ 0.944,  0.   ,  0.041,  0.015],
        # [ 0.038,  0.962,  0.   ,  0.   ],
        # [ 0.   ,  0.023,  0.906,  0.071],
        # [ 0.   ,  0.018,  0.018,  0.964]])

        # counting RD
        # A = np.array([[ 0.432,  0.215,  0.141,  0.212],
        # [ 0.196,  0.346,  0.161,  0.297],
        # [ 0.184,  0.21 ,  0.317,  0.288],
        # [ 0.132,  0.21 ,  0.15 ,  0.507]])

        A = hmm_util.getAmatrix(TextGrid_Directory, CsvDirectory, N)

    A = hmm_util.normalize(A)
    if B_norm:
        B = hmm_util.normalize(B) # or not normalize it

    # accuracy after norm
    prediction_orignal = np.argmax(B,0)
    accuracy_norm = sum(y==prediction_orignal)*1.0/len(y)

    # F-score
    # FSCORE = f1_score(y, prediction_orignal,average='macro')

    if train:
        newPi, newA, newB, logP_list = myhmm.learn_HMM(pi, A, B, iterlimit=1000, threshold=0.0001)
    # Viterbi to compute phi/most likely state sequence
    if not train:
        newA = A
        newB = B
        newPi = pi

    updated_accuracy, best_sequence, FSCORE = myhmm.test_HMM_viterbi(M, N, newPi, newA, newB, y, lamb)

    print "lambda:", lamb
    print "NN accuracy, NN_FSCORE, update accuracy, update F-score:", accuracy_original, FSCORE_original, updated_accuracy, FSCORE
    # print "F-SCORE:", FSCORE
    # pdb.set_trace()

    # plt = plotconf(newA, xlab=['CRY','FUS','LAU','BAB'], ylab=['CRY','FUS','LAU','BAB'], title="A Matrix")
    # # plt = plotconf(newB, title="B Matrix")
    # plt.show()

    # # plt.plot(logP_list)
    # # plt.xlabel('iteration number')
    # # plt.ylabel('log probability')
    # # plt.title('log probability trend')
    # # plt.show()

    # pdb.set_trace()
    # sys.stdout = orig_stdout
    # f.close()
if __name__ == "__main__":
    main()
