batch_size = 16
time_steps = 3
activation_fn = tf.sigmoid
num_input_nodes = 2
num_nodes_per_hidden_layer = [3, 2, 4]
num_hidden_layers = len(num_nodes_per_hidden_layer)
num_output_nodes = 2
num_layers = 1 + num_hidden_layers + 1
num_nodes_per_layer = \
    [num_input_nodes] + num_nodes_per_hidden_layer + [num_output_nodes]


tf.reset_default_graph()

with tf.variable_scope('model'):
    # Feedforward interlayer weights
    V = [tf.get_variable('V_{}_{}'.format(j, i),
        # XXX Is this shape the correct order?
        shape=[num_nodes_per_layer[i], num_nodes_per_layer[j]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for i,j in ((0,1), (1,2), (2,3), (3,4))]

    # Recurrent interlayer weights
    W = [tf.get_variable('W_{}_{}'.format(j, i),
        # XXX Is this shape the correct order?
        shape=[num_nodes_per_layer[i], num_nodes_per_layer[j]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for j, i in itertools.product(range(1, num_layers), repeat=2)]

    b = [tf.get_variable('b_{}'.format(i),
        shape=[num_nodes_per_layer[i]],
        initializer=tf.contrib.layers.xavier_initializer()
    ) for i in range(1, num_layers)]

    y = tf.get_variable('y',
        shape=[time_steps, batch_size, num_output_nodes],
        initializer=tf.constant_initializer(0.),
    )


def build_recurrent_network():
    X = tf.placeholder(dtype=tf.int32,
        shape=[time_steps, batch_size, num_input_nodes])

    states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
              for i in range(1, num_layers)]

    new_states = [tf.zeros([batch_size, num_nodes_per_layer[i]])
                  for i in range(1, num_layers)]

    # Expecting y of shape=[time_steps, batch_size, num_output_nodes]
    for t in range(time_steps):
        # Assumes shape=[time_steps, batch_size, num_input_nodes]
        current_inputs = X[t]

        for i in range(1, num_layers):
            with tf.variable_scope('model', reuse=True):
                V = tf.get_variable('V_{}_{}'.format(i - 1, i))
                b = tf.get_variable('b_{}'.format(i))

            recurrent_term = tf.zeros([batch_size, num_nodes_per_layer[i]])

            for j in range(1, num_layers):
                with tf.variable_scope('model', reuse=True):
                    W = tf.get_variable('W_{}_{}'.format(j, i))

                # XXX Do I need to worry about batch_size dimension in matmul, add?
                recurrent_term += tf.matmul(states[j], W)

            activation = activation_fn(
                # XXX Do I need to worry about batch_size dimension in matmul, add?
                tf.matmul(current_inputs, V) + recurrent_term + b
            )

            current_inputs = activation
            new_states[i] = activation

        y[t] = activation
        states = new_states

    return y


y = build_recurrent_network()