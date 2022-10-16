# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import itertools
import random
from pprint import pprint


batch_size = 16
time_steps = 4 # Note this needs to be whatever I pass to create_samples + 1
num_input_nodes = 2
num_nodes_per_hidden_layer = [3, 2, 4]
num_hidden_layers = len(num_nodes_per_hidden_layer)
num_output_nodes = 2
num_layers = 1 + num_hidden_layers + 1
num_nodes_per_layer = \
    [num_input_nodes] + num_nodes_per_hidden_layer + [num_output_nodes]
num_target_nodes = 1
epochs = 50


# activation_fn = tf.sigmoid
# XXX Note i am not clear how the gradient of tf.round works.
# XXX That means i cannot train it yet
# activation_fn = lambda x: tf.round(tf.sigmoid(x))
# Should I give it a name (ops(?).name_scope?)
# XXX Hopefully it works that i override only the Round gradient
# XXX (tf.sigmoid's gradient should be fine)
def step_fn(x):
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        output = tf.round(tf.sigmoid(x))
    return output

activation_fn = step_fn

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
    recurrent_terms = []
    kernels = []

    states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
              for i in range(0, num_layers)]

    new_states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
                  for i in range(0, num_layers)]

    # Expecting y of shape=[time_steps, batch_size, num_output_nodes]
    for t in range(time_steps):
        # Assumes shape=[time_steps, batch_size, num_input_nodes]
        current_inputs = X[t]

        recurrent_terms_layers = []
        kernels_layers = []

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

            recurrent_terms_layers.append(recurrent_term)

            kernel = tf.matmul(current_inputs, V) + recurrent_term + b

            kernels_layers.append(kernel)

            activation = activation_fn(
                # XXX Do I need to worry about batch_size dimension in matmul, add?
                kernel
            )

            current_inputs = activation
            new_states[i] = activation

        recurrent_terms.append(recurrent_terms_layers)
        kernels.append(kernels_layers)
        y.append(activation)
        states = new_states

    y = tf.stack(y)

    return y, kernels, recurrent_terms


# shape=[time_steps, batch_size, num_output_nodes]
y, kernels, recurrent_terms = build_recurrent_network()

print('y.shape', y.shape)

y_reshaped = tf.reshape(y, [-1, num_output_nodes])

print('y_reshaped.shape', y_reshaped.shape)

# Only take the first output node (we don't train on the carry)
y_output = y_reshaped[:, 0]

print('y_output.shape', y_output.shape)

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
logits_reshaped = tf.reshape(logits, [-1, num_target_nodes])

y_target = tf.placeholder(dtype=tf.int32,
    shape=[time_steps, batch_size, num_target_nodes])

# y_target_reshaped = tf.reshape(y_target, [-1, num_target_nodes])
y_target_reshaped = tf.reshape(y_target, [-1])

# shape=[64,1]
print('y_target_reshaped.shape', y_target_reshaped.shape)

# y_target_reshaped_distributed = tf.concat(
#     [1 - y_target_reshaped, y_target_reshaped], 
#     1
# )

# XXX Maybe simply compute the proability of the counter event here?
# XXX I mean it's trivial but if 
# XXX So the probability that i already have in y_target_reshaped is the prob that y is 1, not 0
# XXX That is the second class afaik
# XXX That means that i need to *pre*pend 1 - that probability
print('logits_reshaped.shape', logits_reshaped.shape)
print('(1-logits_reshaped).shape', (1-logits_reshaped).shape)

logits_reshaped_distributed = tf.concat(
    [1 - logits_reshaped, logits_reshaped], 
    1
)

print('logits_reshaped_distributed.shape', logits_reshaped_distributed.shape)

# optimization
# XXX Using sparse_softmax_cross_entropy_with_logits now
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_reshaped_distributed, labels=y_target_reshaped
)
# XXX Need cross entropy without built-in softmax
# losses = -tf.reduce_sum(
#     y_target_reshaped_distributed * tf.log(logits_reshaped_distributed), 
#     reduction_indices=[1]
# )

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

                # print('xs.shape', xs.shape)
                # print('ys.shape', ys.shape)
                # print('xs', xs)
                # print('ys', ys)

                _, \
                V_, \
                recurrent_terms_, \
                kernels_, \
                logits_reshaped_, \
                y_target_reshaped_, \
                losses_, \
                loss_ \
                = sess.run([
                        train_op,
                        V,
                        recurrent_terms,
                        kernels,
                        logits_reshaped,
                        y_target_reshaped,
                        losses,
                        loss,
                    ],
                    feed_dict = {
                        X : xs,
                        y_target : ys
                    })

                if j == 0:
                    print(
                        xs.shape,
                        # These 2 are apparently time_steps before batch_size
                        # That's why I am currently only printing 4 examples
                        # len(recurrent_terms_), -> 4
                        # len(kernels_), -> 4
                        # Now printing shape of first element of both
                        # I would expect ...
                        # Ah or are these lists of 1 tensor for each *layer*?
                        # Nope, contain list (again)
                        # len(recurrent_terms_[0]),
                        # len(kernels_[0]),
                        # Ok these are length of 4 (again)
                        # Now printing shape of first element of first element
                        # So in total their size is (time_steps, layers, batch_size, num_nodes_per_layer)
                        # Could tf.stack, ah no bc of different sizes of layers
                        recurrent_terms_[0][0].shape, # -> (16,3) -> (batch_size, num_nodes_per_layer[i])
                        kernels_[0][0].shape, # -> (16,3) -> (batch_size, num_nodes_per_layer[i])
                        logits_reshaped_.shape,
                        y_target_reshaped_.shape,
                        losses_.shape
                    )

                    raise Exception

                    print('V_', V_)
                    print('xs, '
                          'recurrent_terms_, '
                          'kernels_, '
                          'logits_reshaped_, '
                          'y_target_reshaped_, '
                          'losses_')

                    pprint(
                        list(zip(
                            xs.reshape([-1, num_input_nodes]).tolist(),
                            # [x.tolist() for x in recurrent_terms_],
                            recurrent_terms_,   
                            # [x.tolist() for x in kernels_],
                            kernels_,
                            logits_reshaped_.tolist(),
                            y_target_reshaped_.tolist(),
                            losses_.tolist())))
                    print('y_target_reshaped_', y_target_reshaped_)
                    print('losses', losses)
                    print('loss_',loss_)

                raise Exception

                train_loss += loss_
            print('[{}] loss : {}'.format(i,train_loss/1000))
            train_loss = 0
    except KeyboardInterrupt:
        print('interrupted by user at ' + str(i))
        #
        # training ends here;
        #  save checkpoint
        saver = tf.train.Saver()
        saver.save(sess, ckpt_path + 'ilrnn1.ckpt', global_step=i)

