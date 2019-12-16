"""
Usage:
python train_prosody.py \
--filenames=balanced_tfrecords_21/train.tfrecord \
--batch_size=30 --num_epochs=1200 \
--logDir='ckpt_21_1layer/' --restore=False --num_classes=4
--prosody_or_fbank='prosody'
"""
import os
from tensorflow import logging
import tensorflow as tf
import read_data
import pdb
import CNNmodel

flags = tf.app.flags
flags.DEFINE_string(
    'filenames', None,
    'Path to *tfrecord') # 'tfrecords/output001.wav.tfrecord'
# each example has 100 frames -> 10x100 frame examples
flags.DEFINE_integer("batch_size", 10, "How many examples to process per batch for training.")
flags.DEFINE_integer("num_epochs", 1, "How many examples to process per batch for training.")
flags.DEFINE_string(
    'logDir', None,
    'Path to ckpt') # 'ckpt/'
flags.DEFINE_bool(
    'restore', False,
    'If you want to restore the ckpt')
flags.DEFINE_integer(
    'num_classes', 4,
    'The number of classes')
flags.DEFINE_string(
    'prosody_or_fbank', None,
    'choose one') # 'ckpt/'
FLAGS = flags.FLAGS

def main(_):
    # Logging the version.
    logging.set_verbosity(tf.logging.INFO)
    if FLAGS.filenames:
        filenames= FLAGS.filenames
    if FLAGS.batch_size:
        batch_size= FLAGS.batch_size
    if FLAGS.num_epochs:
        num_epochs= FLAGS.num_epochs

    # Load data placeholders of [batch_size, 100, 64]
    images, labels, filenames = read_data.loadembedding(filenames, batch_size=batch_size, num_epochs=num_epochs, feature_type=FLAGS.prosody_or_fbank)

    # build model graph
    model = CNNmodel.CNNmodel(inputs=images,labels=labels,num_classes=FLAGS.num_classes,is_train=True, feature_type=FLAGS.prosody_or_fbank)

    # Train - or for testing tensor purpose
    # train_op = model._trainop
    # loss = model._loss
    # predictions = model._predictions
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in xrange(5): # for each batch
    #         _, img, lab, los, pred, logits, new_lab, accu = sess.run([train_op, images, labels, loss, predictions, model._logits, model.new_label, model.accuracy]) # size of [batch_size,height,width] - each i returns one batch of examples
    #         logging.info("training step " + str(i) + " | Loss: " + ("%.2f" % los))
    #         pdb.set_trace()
    #     coord.request_stop()
    #     coord.join(threads)

    if FLAGS.restore:
        model.load_checkpoint(FLAGS.logDir)
    model.train(FLAGS.logDir)

if __name__ == '__main__':
    tf.app.run()
