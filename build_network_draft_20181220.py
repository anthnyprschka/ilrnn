import tensorflow as tf


batch_size = 16
activation = tf.sigmoid
num_input_nodes = 2
num_nodes_per_hidden_layer = np.array([3, 2, 4])
num_hidden_layers = num_nodes_per_hidden_layer.shape[0]
num_output_nodes = 2
num_layers = num_hidden_layers + 1
num_nodes_per_layer = \
    num_nodes_per_hidden_layer + [num_output_nodes]



def build_recurrent_term(i, j, recurrent_term, states):
    recurrent_term += tf.matmul(states[j], W[j][i])

    return (i, j+1, recurrent_term, states)


def build_layer(i, inputs, states, new_states):
    j = tf.constant(0.)
    cond = lambda _, j: tf.less(j, num_layers)

    V = tf.get_variable('V_{{{}, {}}}'.format(j, i),
        shape=[num_nodes_per_layer[i], num_nodes_per_layer[j]],
        initializer=tf.xavier
    )
    b = tf.get_variable('b_{}'.format(i),
        shape=[num_nodes_per_layer[i]],
        initializer=tf.xavier
    )
    recurrent_term = tf.get_variable('r_{}'.format(i),
        shape=[batch_size, num_nodes_per_layer[i]],
        initializer=tf.zeros
    )

    _, _, recurrent_term = tf.while_loop(
        cond,
        build_recurrent_term,
        [i, j, recurrent_term, states]
    )

    activation = activation(
        # XXX Do I need to worry about batch_size dimension in matmul, add?
        tf.matmul(inputs, V) + recurrent_term + b
    )

    new_states[i] = activation

    return (i + 1, activation, states, new_states)


def build_feed_forward_network(t, inputs, states, outputs):
    i = tf.constant(0.)
    cond = lambda i: tf.less(i, num_layers)

    # Assumes shape=[time_steps, batch_size, num_input_nodes]
    current_inputs = inputs[t]

    _, output, _, states = tf.while_loop(
        cond,
        build_layer,
        [i, current_inputs, states, states]
    )

    outputs[t] = output

    return (t + 1, inputs, states, outputs)


def build_recurrent_network():
    tf.reset_default_graph()

    t = tf.constant(0.)
    cond = lambda t: tf.less(t, time_steps)

    X = tf.placeholder('X',
        shape=[time_steps, batch_size, num_input_nodes])

    initial_state = [tf.get_variable(
                        'a_{{}}^{t-1}'.format(i),
                        shape=[batch_size, num_nodes_per_layer[i]],
                        initializer=tf.zeros())
                     for i in range(num_layers)]

    y = tf.get_variable('y',
        shape=[time_steps, batch_size, num_output_nodes],
        initializer=tf.zeros,
    )

    # Expecting y of shape=[time_steps, batch_size, num_output_nodes]
    _, _, _, y = tf.while_loop(
        cond,
        build_feed_forward_network,
        [t, X, initial_state, y]

    )

    return y


y = build_recurrent_network()


y_hat = tf.placeholder('y_hat',
    shape=[time_steps, batch_size, num_target_nodes]
)

# XXX Needs to be summed across time steps
loss = tf.cross_entropy(y, y_hat)


def create_data():
    return


def generate_batches(data, batch_size):
    while True:
        X = np.zeros([batch_size, time_steps, num_input_nodes])
        y = np.zeros([batch_size, time_steps, num_target_nodes])

        for x in range(batch_size):
            random_index = random.randint(0, len(data))

            X[x] = data[random_index][0]
            y[x] = data[random_index][1]
        # shape=[time_steps, batch_size, num_input_nodes]
        # shape=[time_steps, batch_size, num_target_nodes]
        X = X.transpose()
        y = y.transpose()

        yield X, y


data = generate_data()
batch_generator = generate_batches(data, batch_size)


num_epochs = 5


with tf.default_session() as sess:
    for num_batch in range(num_epochs * int(len(data) / batch_size)):
        X, y = batch_generator.next()

        loss, _ = sess.run([loss, optimizer],
            feed_dict={'X': X, 'y': y}
        )

        if num_batch % 100 == 0:
            print('Loss after batch {}: {}'.format(num_batch, loss))

