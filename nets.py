import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb
import params

def create_prosody_model(input_batch,num_classes,labels_batch,is_training):
    input_batch = tf.reshape(input_batch, [-1, params.NUM_FEATURES]) #[batch,20]
    # input_batch = tf.reshape(input_batch, [-1, params.NUM_FEATURE_SINGLE]) #for one feature case

    # net = slim.fully_connected(
    #   input_batch, 128, activation_fn=tf.nn.relu)
    # net = slim.fully_connected(
    #   net, 256, activation_fn=tf.nn.relu)
    net = input_batch
    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='predictions')
    return logits #[batch_size, num_classes]

def create_fbank_model(input_batch,num_classes,labels_batch,is_training):
    dim_list = input_batch.get_shape().as_list() # [batch,96,64]
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        biases_initializer=tf.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=is_training), \
         slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1, padding='SAME'), \
         slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=4, padding='SAME'):

        net = tf.reshape(input_batch, [-1, dim_list[1], dim_list[2], 1]) #[batch,96,64,1]
        net = slim.conv2d(net, 8, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1') # batch, 48, 32, 8
        net = slim.conv2d(net, 16, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2') # batch, 24, 16, 16
        # net = slim.conv2d(net, 16, scope='conv3')
        # net = slim.max_pool2d(net, scope='pool3') # batch, 12, 8, 32
        # net = slim.conv2d(net, 64, scope='conv4')
        # net = slim.max_pool2d(net, scope='pool4') # batch, 6, 4, 64
        # pdb.set_trace()

        net = slim.flatten(net) # batch
        # net = slim.fully_connected(net, 128, scope='fc1')
        net = slim.fully_connected(net, 64, scope='fc2')
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='predictions')
        if not is_training:
            logits = tf.reduce_sum(logits,0)
            logits = tf.reshape(logits,[-1,num_classes])
        return logits #[batch_size, num_classes]
