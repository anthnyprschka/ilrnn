# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import itertools
import random
from pprint import pprint


batch_size = 16
time_steps = 4 # Note this needs to be whatever I pass to create_samples + 1
activation_fn = tf.sigmoid
num_input_nodes = 2
num_nodes_per_hidden_layer = [3, 2, 4]
num_hidden_layers = len(num_nodes_per_hidden_layer)
num_output_nodes = 2
num_layers = 1 + num_hidden_layers + 1
num_nodes_per_layer = \
    [num_input_nodes] + num_nodes_per_hidden_layer + [num_output_nodes]
num_target_nodes = 1
epochs = 50


tf.reset_default_graph()


with tf.variable_scope('model') as scope:
    # Feedforward interlayer weights
    V = [tf.get_variable(u'V_{}_{}'.format(i, j),
        # XXX Is this shape the correct order?
        shape=[num_nodes_per_layer[i], num_nodes_per_layer[j]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for i,j in ((0,1), (1,2), (2,3), (3,4))]

    # Recurrent interlayer weights
    W = [tf.get_variable(u'W_{}_{}'.format(j, i),
        shape=[num_nodes_per_layer[j], num_nodes_per_layer[i]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for j, i in itertools.product(range(1, num_layers), repeat=2)]

    b = [tf.get_variable(u'b_{}'.format(i),
        shape=[num_nodes_per_layer[i]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for i in range(1, num_layers)]

    # y = tf.get_variable('y',
    #     shape=[time_steps, batch_size, num_output_nodes],
    #     initializer=tf.constant_initializer(0.),
    # )

X = tf.placeholder(dtype=tf.float32,
    shape=[time_steps, batch_size, num_input_nodes])

def build_recurrent_network():
    y = []

    states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
              for i in range(0, num_layers)]

    new_states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
                  for i in range(0, num_layers)]

    # Expecting y of shape=[time_steps, batch_size, num_output_nodes]
    for t in range(time_steps):
        # Assumes shape=[time_steps, batch_size, num_input_nodes]
        current_inputs = X[t]

        for i in range(1, num_layers):
            with tf.variable_scope(scope, reuse=True):
                V = tf.get_variable('V_{}_{}'.format(i - 1, i))
                b = tf.get_variable('b_{}'.format(i))

            recurrent_term = tf.zeros([batch_size, num_nodes_per_layer[i]])

            for j in range(1, num_layers):
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable('W_{}_{}'.format(j, i))

                # XXX Do I need to worry about batch_size dimension in matmul, add?
                recurrent_term += tf.matmul(states[j], W)

            activation = activation_fn(
                # XXX Do I need to worry about batch_size dimension in matmul, add?
                tf.matmul(current_inputs, V) + recurrent_term + b
            )

            current_inputs = activation
            new_states[i] = activation

        y.append(activation)
        states = new_states

    y = tf.stack(y)

    return y


# shape=[time_steps, batch_size, num_output_nodes]
y = build_recurrent_network()

y_reshaped = tf.reshape(y, [-1, num_output_nodes])

# Only take the first output node (we don't train on the carry)
y_output = y_reshaped[:, 0]

# shape=[ num_samples, time_steps, |[x0, x1, y]| ]
# Still need to normalize/pad to one value for time_steps
# np.array(for x in create_samples(3))
# Number of digits (f.e. 3), digits \in [0.0, 1.0]
# 0, 1, 10, 11, 100, 101, 111, 1000
# [0, 0], [1, 0], [0, 1], [1, 1], []
# [0000, 0000], [0001, 0000], [0000, 0001], [0001, 0001]
# Note we are interpreting the binary numbers generated here in reverse order
# Currently not adding the the 3 timestep if its y would be 1
def create_samples(num_digits):
    samples = []

    for sequence in itertools.product([0, 1], repeat=2*num_digits):

        sample = []
        carry = 0

        for time_step in range(num_digits):

            x_1 = sequence[time_step]
            x_2 = sequence[num_digits + time_step]

            if not carry:
                y_1 = x_1 ^ x_2
            else:
                y_1 = ~(x_1 ^ x_2) & ((1<<1)-1) # ???

            # Ideally do this using bitwise operations
            if carry + x_1 + x_2 > 1:
                carry = 1
            else:
                carry = 0

            sample.append([x_1, x_2, y_1])


        if carry:
            sample.append([0,0,1])

        samples.append(sample)

    return samples


data = np.array([x + [[0,0,0]] if len(x) < 4 else x
                 for x in create_samples(3)])

# shape=[ time_steps, num_samples, |[x0, x1, y]| ]
data = np.transpose(data, (1, 0, 2))

# print('data.shape', data.shape)
# print('data[0]', data[0].shape)

def generate_batches(data, batch_size):
    while True:
        X = np.zeros([time_steps, batch_size, num_input_nodes])
        y = np.zeros([time_steps, batch_size, num_target_nodes])

        for x in range(batch_size):
            random_index = random.randint(0, len(data))

            X[:, x] = data[:, random_index, :2]
            y[:, x] = data[:, random_index, 2:]

        yield X, y


batch_generator = generate_batches(data, batch_size)

# some_sample = batch_generator.next()
# print(zip(some_sample[0], some_sample[1]))

logits = y_output
# shape=[64,]
print('logits.shape', logits.shape)

# predictions = tf.nn.softmax(logits)
logits_reshaped = tf.reshape(logits, [-1, 1])

y_target = tf.placeholder(dtype=tf.float32,
    shape=[time_steps, batch_size, num_target_nodes])

y_target_reshaped = tf.reshape(y_target, [-1, num_target_nodes])

# shape=[64,1]
print('y_target_reshaped.shape', y_target_reshaped.shape)

# optimization
losses = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits_reshaped, labels=y_target_reshaped
)

print('losses.shape', losses.shape)

loss = tf.reduce_mean(losses)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)


# training session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_loss = 0
    try:
        for i in range(epochs):
            for j in range(1000):
                xs, ys = batch_generator.next()

                print('xs.shape', xs.shape)
                print('ys.shape', ys.shape)
                print('xs', xs)
                print('ys', ys)

                _, V_, logits_reshaped_, y_target_reshaped_, train_loss_, losses_ = sess.run(
                    [train_op, V, logits_reshaped, y_target_reshaped, loss, losses],
                    feed_dict = {
                        X : xs,
                        y_target : ys
                    })

                print('V_', V_)
                print('logits_reshaped_, y_target_reshaped_, losses_')
                pprint(list(zip(logits_reshaped_.tolist(),
                                y_target_reshaped_.tolist(),
                                losses_.tolist())))
                # print('y_target_reshaped_', y_target_reshaped_)
                # print('losses', losses)
                print('train_loss_',train_loss_)

                raise Exception

                train_loss += train_loss_
            print('[{}] loss : {}'.format(i,train_loss/1000))
            train_loss = 0
    except KeyboardInterrupt:
        print('interrupted by user at ' + str(i))
        #
        # training ends here;
        #  save checkpoint
        saver = tf.train.Saver()
        saver.save(sess, ckpt_path + 'ilrnn1.ckpt', global_step=i)