import tensorflow as tf 


def build_model(time_steps):
	with tf.variable_scope('lower_layer', reuse=None):
		# ...

	with tf.variable_scope('lower_layer', reuse=True):
		# ...


with tf.variable_scope('top_layer', reuse=None):
	time_steps_train = 3

	y_train = build_model(time_steps_train)


with tf.variable_scope('top_layer', reuse=True):
	time_steps_eval = 5

	y_eval = build_model(time_steps_eval)