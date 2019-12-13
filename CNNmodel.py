import tensorflow as tf
import nets
import pdb
import params 
slim = tf.contrib.slim

class CNNmodel(object):
	# connect to input, target placeholders 
	def __init__(self,inputs,labels,num_classes,feature_type,sess=None,is_train=True):
		self._istrain=is_train

		if sess is None:
			self._sess = tf.Session()
		else:
			self._sess = sess
		self._feature_type=feature_type
		self._inputs=inputs
		self._labels=labels
		self._num_classes=num_classes
		self._predictions=self._predictions()
		self._init_fn = None
		if self._istrain:
			self._scalars_to_log=[]
			self._loss=self._loss() 
			self._trainop=self._create_trainop()
			self._create_summaries()

		print('Input Size:',self._inputs)
		print('Output Size:',self._labels)

	def _predictions(self):
		"""get the prediction labels"""
		input_raw=self._inputs #[batch,1,20]
		labels_batch=self._labels 
		num_classes=self._num_classes
		# feature_dim = len(input_raw.get_shape()) - 1
		# input_batch = tf.nn.l2_normalize(input_raw, feature_dim) # [batch_size,100,64]
		
		# for debug use
		# self.new_img=tf.reshape(input_raw, [-1, params.NUM_FEATURE_SINGLE]) #for one feature case
		# self.new_img=tf.reshape(input_raw, [-1, params.NUM_FEATURES]) 
		if self._feature_type=='prosody':
			predictions = nets.create_prosody_model(input_batch=input_raw, num_classes=num_classes, labels_batch=labels_batch, is_training=self._istrain)
		else:
			predictions = nets.create_fbank_model(input_batch=input_raw, num_classes=num_classes, labels_batch=labels_batch, is_training=self._istrain)
		# self.pred_prev = pred_prev
		# self.pred_new = predictions
		self._logits = tf.nn.softmax(predictions)
		if not self._istrain: #test mode 
			predictions = tf.argmax(tf.nn.softmax(predictions),1) # now only [1,4] for the whole segment 
		return predictions

	def _CrossEntropyLoss(self, predictions, labels, **unused_params):
		epsilon = 10e-6
		float_labels = tf.cast(labels, tf.float32)
		cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
			1 - float_labels) * tf.log(1 - predictions + epsilon)
		cross_entropy_loss = tf.negative(cross_entropy_loss)
		return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

	def _loss(self):
		predictions = self._predictions #[batch,num_classes] 
		labels = self._labels #[batch,1]
		labels = tf.one_hot(labels,self._num_classes) # [batch,num_classes]
		self.new_label = labels # FOR DEBUG
		# loss = self._CrossEntropyLoss(predictions,labels)		
		loss = tf.losses.softmax_cross_entropy(labels,predictions)
		self._scalars_to_log.append(('CrossEntropyLoss', loss))
		tf.losses.add_loss(loss)

		# eval
		if self._istrain:
			correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(predictions), 1), tf.argmax(labels, 1))
		if not self._istrain:
			correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		self.accuracy = accuracy  # FOR DEBUG
		self._scalars_to_log.append(('Accuracy', accuracy))
		return loss 

	def _create_trainop(self):
		# global_step = slim.get_or_create_global_step()
		# optimizer = tf.train.AdamOptimizer(params.LEARNING_RATE)
		optimizer = tf.train.GradientDescentOptimizer(params.LEARNING_RATE)
		self.optimizer = optimizer 
		total_loss = tf.losses.get_total_loss()
		train_op = slim.learning.create_train_op(total_loss, optimizer, clip_gradient_norm=4)
		return train_op
	
	def _create_summaries(self):
		for var_name, var in self._scalars_to_log:
		            tf.summary.scalar(var_name, var)

	def train(self,logDir):
		slim.learning.train(self._trainop,
							logDir,
							init_fn = self._init_fn,
							save_summaries_secs= 60, #30, 
							save_interval_secs= 60 #30 #1 * 60, #save model 
							)

	def evaluation(self,log_dir,ckptdir):
		latest_ckpt=tf.train.latest_checkpoint(ckptdir) 

		summary_vars = dict(self._scalars_to_log)
		metrics = {'eval_accuracy':slim.metrics.streaming_mean(summary_vars['Accuracy'])}	
		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metrics)
		print("########################")
		print(summary_vars['Accuracy'])
		print(names_to_updates.values())
		print("########################")
		slim.evaluation.evaluation_loop('',
										latest_ckpt,
										log_dir,
										num_evals=1, #num of batches to evaluate 
										eval_interval_secs=30,
										init_fn=self._init_fn,
										eval_op=list(names_to_updates.values())
										)

	def load_checkpoint(self, checkpoint_dir):
		# pdb.set_trace()
		latest_ckpt=tf.train.latest_checkpoint(checkpoint_dir)  
		variables_to_restore = slim.get_model_variables()
		init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
			latest_ckpt, variables_to_restore)
		self._init_fn=lambda :self._sess.run(init_assign_op, init_feed_dict) # Create an initial assignment function.
		pass
