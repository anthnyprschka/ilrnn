# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import itertools
import random
from pprint import pprint
import time


batch_size = 16

time_steps = 4
time_steps_train = 4
time_steps_eval = 5

num_input_nodes = 2
num_nodes_per_hidden_layer = [3, 2, 4]
num_hidden_layers = len(num_nodes_per_hidden_layer)
num_output_nodes = 2
num_layers = 1 + num_hidden_layers + 1
num_nodes_per_layer = \
    [num_input_nodes] + num_nodes_per_hidden_layer + [num_output_nodes]
num_target_nodes = 1
epochs = 1000


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
    V = [tf.constant([
             [1.0, -1.0, 1.0],
             [-1.0, 1.0, 1.0]
         ], name='V_{}_{}'.format(0,1)),
         tf.constant([
             [1.0, 0.0],
             [1.0, 0.0],
             [0.0, 1.0]
         ], name='V_{}_{}'.format(1,2)),
         tf.constant([
             [1.0, -1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]
         ], name='V_{}_{}'.format(2,3)),
         tf.constant([
             [1.0, 0.0],
             [1.0, 0.0],
             [0.0, 1.0],
             [0.0, 1.0]
         ], name='V_{}_{}'.format(3,4))]


    print('V', V)

    # Recurrent interlayer weights
    W = [tf.zeros([
             num_nodes_per_layer[1], num_nodes_per_layer[1]
         ], name='W_{}_{}'.format(1,1)),
         tf.zeros([
             num_nodes_per_layer[1], num_nodes_per_layer[2]
         ], name='W_{}_{}'.format(1,2)),
         tf.zeros([
             num_nodes_per_layer[1], num_nodes_per_layer[3]
         ], name='W_{}_{}'.format(1,3)),
         tf.zeros([
             num_nodes_per_layer[1], num_nodes_per_layer[4]
         ], name='W_{}_{}'.format(1,4)),
         tf.zeros([
             num_nodes_per_layer[2], num_nodes_per_layer[1]
         ], name='W_{}_{}'.format(2,1)),
         tf.zeros([
             num_nodes_per_layer[2], num_nodes_per_layer[2]
         ], name='W_{}_{}'.format(2,2)),
         tf.zeros([
             num_nodes_per_layer[2], num_nodes_per_layer[3]
         ], name='W_{}_{}'.format(2,3)),
         tf.zeros([
             num_nodes_per_layer[2], num_nodes_per_layer[4]
         ], name='W_{}_{}'.format(2,4)),
         tf.zeros([
             num_nodes_per_layer[3], num_nodes_per_layer[1]
         ], name='W_{}_{}'.format(3,1)),
         tf.zeros([
             num_nodes_per_layer[3], num_nodes_per_layer[2]
         ], name='W_{}_{}'.format(3,2)),
         tf.zeros([
             num_nodes_per_layer[3], num_nodes_per_layer[3]
         ], name='W_{}_{}'.format(3,3)),
         tf.zeros([
             num_nodes_per_layer[3], num_nodes_per_layer[4]
         ], name='W_{}_{}'.format(3,4)),
         tf.zeros([
             num_nodes_per_layer[4], num_nodes_per_layer[1]
         ], name='W_{}_{}'.format(4,1)),
         tf.zeros([
             num_nodes_per_layer[4], num_nodes_per_layer[2]
         ], name='W_{}_{}'.format(4,2)),
         tf.constant([
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 1.0, 0.0]
         ], name='W_{}_{}'.format(4,3)),
         tf.zeros([
            num_nodes_per_layer[4], num_nodes_per_layer[2]
         ], name='W_{}_{}'.format(4,4))]

    # Biases
    b = [tf.constant([0.0, 0.0, -1.0], name='b_{}'.format(1)),
         tf.constant([0.0, 0.0], name='b_{}'.format(2)),
         tf.constant([0.0, 0.0, -1.0, 0.0], name='b_{}'.format(3)),
         tf.constant([0.0, 0.0], name='b_{}'.format(4))]

    # y = tf.get_variable('y',
    #     shape=[time_steps, batch_size, num_output_nodes],
    #     initializer=tf.constant_initializer(0.),
    # )

X = tf.placeholder(dtype=tf.float32,
    # shape=[time_steps, batch_size, num_input_nodes])
    # shape=[time_steps, None, num_input_nodes])
    shape=[None, None, num_input_nodes])

# So würde ich aber den array außerhalb des placeholders haben
# sodass ich im feed dict keinen array passen könnte, oder?
# states = [tf.placeholder(dtype=tf.float32,
#     shape=[None, num_nodes_per_layer[i]]
# ) for i in range(0, num_layers)]

# new_states = [tf.placeholder(dtype=tf.float32,
#     shape=[None, num_nodes_per_layer[i]]
# ) for i in range(0, num_layers)]

# # XXX
# recurrent_terms = [tf.placeholder(dtype=tf.float32,
#     # XXX Shape
#     shape=[None, num_nodes_per_layer[i]],
# )
# for i in range(0, num_layers) 
# for j in range(0, num_layers)
# for k in range(0, time_steps)]


def build_recurrent_network(batch_size, time_steps):
    y = []
    recurrent_terms = []
    kernels = []

    # XXX Can I put tf.zeros of unknown size of one dimension?
    # XXX Nope but I can use tf.zeros_like
    # XXX Or I could use variables that are initialized to zero...
    states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
              for i in range(0, num_layers)]
    # states = [tf.get_variable(u's_{}'.format(i),
    #     shape=[None, num_nodes_per_layer[i]],
    #     initializer=tf.constant_initializer(0.)
    # ) for i in range(0, num_layers)]

    new_states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
                  for i in range(0, num_layers)]
    # new_states = [tf.get_variable(u's_{}_new'.format(i),
    #     shape=[None, num_nodes_per_layer[i]],
    #     initializer=tf.constant_initializer(0.)
    # ) for i in range(0, num_layers)]

    # Expecting y of shape=[time_steps, batch_size, num_output_nodes]
    for t in range(time_steps):
        # Assumes shape=[time_steps, batch_size, num_input_nodes]
        current_inputs = X[t]

        recurrent_terms_layers = []
        kernels_layers = []

        # Note I am starting with 1, not 0
        for i in range(1, num_layers):
            with tf.variable_scope(scope, reuse=True):
                V = tf.get_default_graph().get_tensor_by_name('model/V_{}_{}:0'.format(i - 1, i))
                b = tf.get_default_graph().get_tensor_by_name('model/b_{}:0'.format(i))

            recurrent_term = tf.zeros([batch_size, num_nodes_per_layer[i]])

            for j in range(1, num_layers):
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_default_graph().get_tensor_by_name('model/W_{}_{}:0'.format(j, i))

                # XXX Do I need to worry about batch_size dimension in matmul, add?
                recurrent_term += tf.matmul(states[j], W)

            recurrent_terms_layers.append(recurrent_term)

            kernel = tf.matmul(current_inputs, V) + recurrent_term + b

            # Trying to figure out why b here has different shape than
            # when I use manual weights
            print('b.shape', b.shape)
            print('kernel.shape', kernel.shape)
            raise Exception

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


with tf.variable_scope('train', reuse=None):
    y, kernels, recurrent_terms = build_recurrent_network(
        batch_size,
        time_steps_train
    )

with tf.variable_scope('train', reuse=True):
    y_eval, _, _ = build_recurrent_network(
        batch_size,
        time_steps_eval
    )    

# with tf.variable_scope('model', reuse=None):
#     # shape=[time_steps, batch_size, num_output_nodes]
#     y, kernels, recurrent_terms = build_recurrent_network(batch_size)

# with tf.variable_scope('model', reuse=True):
#     y_valid = build_recurrent_network(len(data))

print('y.shape', y.shape)

y_reshaped = tf.reshape(y, [-1, num_output_nodes])

print('y_reshaped.shape', y_reshaped.shape)

# Only take the first output node (we don't train on the carry)
y_output = y_reshaped[:, 0]

print('y_output.shape', y_output.shape)


def generate_data(num_seq_length):
    list_of_lists = [[0,1] for x in range(2 * num_seq_length)]

    while True:
        sequence = map(random.choice, list_of_lists)

        sample = []
        carry = 0

        for time_step in range(num_seq_length):

            x_1 = sequence[time_step]
            x_2 = sequence[num_seq_length + time_step]

            if not carry:
                # ^ is the binary XOR operator
                # x_1 and x_2 will be integers of either 0 or 1
                # If I would do a binary operator on integers larger than 0 or 1,
                # then python would first translate it to a binary number and then perform
                # the binary operator on it
                y_1 = x_1 ^ x_2
            else:
                # XNOR
                # XXX What have I done here? What do I need to do?
                # XXX Seems like this also causes shifting/adding overflow after last digit
                y_1 = ~(x_1 ^ x_2) & ((1 << 1) - 1) # ???

            # Ideally do this using bitwise operations
            if carry + x_1 + x_2 > 1:
                carry = 1
            else:
                carry = 0

            sample.append([x_1, x_2, y_1])

        # If carry, either deal with longer sequence length or continue
        if carry:
            # sample.append([0,0,1])
            continue

        # Load to 3-dimensional numpy array of shape [batch_size, time_steps, [x1,x2,y1]]
        sample = np.array([sample])
        # Reshape to [time_steps, batch_size, [x1,x2,y1]]
        sample = np.transpose(sample, (1, 0, 2))

        yield sample


def generate_batches(generate_data, time_steps, batch_size):
    gen = generate_data(time_steps)

    while True:
        X = np.zeros([time_steps, batch_size, num_input_nodes])
        y = np.zeros([time_steps, batch_size, num_target_nodes])

        for x in range(batch_size):
            sample = next(gen)

            X[:, x] = sample[:, 0, :2]
            y[:, x] = sample[:, 0, 2:]

        yield X, y


train_batch_generator = generate_batches(
    generate_data,
    time_steps_train, 
    batch_size
)
eval_batch_generator = generate_batches(
    generate_data, 
    time_steps_eval, 
    batch_size
)

# some_sample = batch_generator.next()
# print(zip(some_sample[0], some_sample[1]))

logits = y_output
# shape=[64,]
print('logits.shape', logits.shape)

# predictions = tf.nn.softmax(logits)
logits_reshaped = tf.reshape(logits, [-1, num_target_nodes])

y_target = tf.placeholder(dtype=tf.int32,
    # shape=[time_steps, batch_size, num_target_nodes])
    # shape=[time_steps, None, num_target_nodes])
    shape=[None, None, num_target_nodes])

# y_target_reshaped = tf.reshape(y_target, [-1, num_target_nodes])
y_target_reshaped = tf.reshape(y_target, [-1])

# Calculate accuracy
labels = y_target_reshaped

print('labels.shape', labels.shape)

predictions = tf.reshape(logits_reshaped, [-1])

print('predictions.shape', predictions.shape)

# Accuracy on training data
train_accuracy = tf.metrics.accuracy(labels, predictions)

# XXX Can definitely be simplified
# Accuracy on eval data with "free" sequence length
predictions_eval = tf.reshape(
    tf.reshape(
        tf.reshape(
            y_eval, 
            [-1, num_output_nodes]
        )[:, 0], 
        [-1, num_target_nodes]
    ), 
    [-1]
)
eval_accuracy = tf.metrics.accuracy(labels, predictions_eval)

print('train_accuracy', train_accuracy)

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
# train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)


# saver = tf.train.Saver()

# training session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_loss = 0
    try:
        for i in range(epochs):

            # Note I am now evaluating *before* the backward pass
            # so that the first eval numbers are without prior training
            # Ok but this shouldn't make a difference of both train *and*
            # eval accuracy are calculated before (or after) first backward pass
            xs_train, ys_train = train_batch_generator.next()

            # print('xs_train.shape', xs_train.shape)
            # print('ys_train.shape', ys_train.shape)

            # Calculate accuracy
            # - on training set
            train_accuracy_, labels_, predictions_ = sess.run(
                [train_accuracy, labels, predictions], 
                feed_dict = {
                    X : xs_train,
                    y_target : ys_train
                }
            )
            # - on validation set
            xs_eval, ys_eval = eval_batch_generator.next()

            eval_accuracy_, _, _ = sess.run(
                [eval_accuracy, labels, predictions], 
                feed_dict = {
                    X : xs_eval,
                    y_target : ys_eval
                }
            )

            # print('labels', labels_)
            # print('predictions', predictions_)

            for j in range(100):
                xs, ys = train_batch_generator.next()

                # print('xs.shape', xs.shape)
                # print('ys.shape', ys.shape)
                # print('xs', xs)
                # print('ys', ys)

                # train_accuracy, labels_, predictions_ = sess.run(
                #     [accuracy, labels, predictions], 
                #     feed_dict = {
                #         X : xs,
                #         y_target : ys
                #     }
                # )
                # print('train_accuracy', train_accuracy)

                # _, \
                V_, \
                recurrent_terms_, \
                kernels_, \
                logits_reshaped_, \
                y_target_reshaped_, \
                losses_, \
                loss_ \
                = sess.run([
                        # train_op,
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
                        y_target : ys,

                        # states: [
                        #     np.zeros((batch_size, num_nodes_per_layer[i])) 
                        #     for i in range(0, num_layers)
                        # ],
                        # new_states: [
                        #     np.zeros((batch_size, num_nodes_per_layer[i])) 
                        #     for i in range(0, num_layers)
                        # ],
                        # recurrent_terms: [
                        #     np.zeros((batch_size, num_nodes_per_layer[i]))
                        #     for i in range(0, num_layers)
                        #     for j in range(0, num_layers)
                        #     for k in range(0, time_steps)
                        # ],
                    })

                # if j == 0:
                #     print(
                #         xs.shape,
                #         # These 2 are apparently time_steps before batch_size
                #         # That's why I am currently only printing 4 examples
                #         # len(recurrent_terms_), -> 4
                #         # len(kernels_), -> 4
                #         # Now printing shape of first element of both
                #         # I would expect ...
                #         # Ah or are these lists of 1 tensor for each *layer*?
                #         # Nope, contain list (again)
                #         # len(recurrent_terms_[0]),
                #         # len(kernels_[0]),
                #         # Ok these are length of 4 (again)
                #         # Now printing shape of first element of first element
                #         # So in total their size is (time_steps, layers, batch_size, num_nodes_per_layer)
                #         # Could tf.stack, ah no bc of different sizes of layers
                #         recurrent_terms_[0][0].shape, # -> (16,3) -> (batch_size, num_nodes_per_layer[i])
                #         kernels_[0][0].shape, # -> (16,3) -> (batch_size, num_nodes_per_layer[i])
                #         logits_reshaped_.shape,
                #         y_target_reshaped_.shape,
                #         losses_.shape
                #     )

                #     # raise Exception

                #     print('V_', V_)
                #     print('xs, '
                #           'recurrent_terms_, '
                #           'kernels_, '
                #           'logits_reshaped_, '
                #           'y_target_reshaped_, '
                #           'losses_')

                #     pprint(
                #         list(zip(
                #             xs.reshape([-1, num_input_nodes]).tolist(),
                #             # [x.tolist() for x in recurrent_terms_],
                #             recurrent_terms_,   
                #             # [x.tolist() for x in kernels_],
                #             kernels_,
                #             logits_reshaped_.tolist(),
                #             y_target_reshaped_.tolist(),
                #             losses_.tolist())))
                #     print('y_target_reshaped_', y_target_reshaped_)
                #     print('losses', losses)
                #     print('loss_',loss_)

                # raise Exception

                train_loss += loss_

            # XXX Add sequence length for both train and eval
            print('[{}] loss : {}, accuracy (train) : {}, accuracy (eval): {}'\
                .format(
                    i,
                    # train_loss / 1000
                    train_loss,
                    train_accuracy_,
                    eval_accuracy_
                )
            )

            train_loss = 0

    except KeyboardInterrupt:
        print('interrupted by user at ' + str(i))
        #
        # training ends here;
        # save checkpoint
        # XXX timestamp including second
        curr_time = time.strftime("%Y%m%d%H%M%S")
        saver.save(sess, './ckpts/ilrnn1_{}.ckpt'.format(curr_time), global_step=i)

    # XXX timestamp including second
    curr_time = time.strftime("%Y%m%d%H%M%S")
    saver.save(sess, './ckpts/ilrnn1_{}.ckpt'.format(curr_time), global_step=i)

# Evaluate

# XXX Should I withhold some samples of same sequence length from training?
# XXX Create dataset (only longer sequence length)
# XXX Does my create_samples function already support producing sequences 
# XXX of size in range (min_size, max_size)?
# XXX I think i need to update it
# XXX I could also just filter, pick from its outputs
# XXX Should samples that overflow (through carry) to length num_digits be contained in the set created by create_samples(num_digits)?


# XXX sess.run only on the predictions
# XXX Do i need to add some postprocessing to the graph for prediction/inference?
# XXX Calculate accuracy
# XXX Jup the insight is that i wanna actually look at accuracy, not the loss

# XXX Plot graph?
# XXX Later: Iteratively increase sequence length and plot accuracy over sequence length
# 






