def add_recurrent_term():
    return

def build_feedforward_layer( i, previous_activation, hidden_states ):
    current_hidden_state = hidden_states[ i ]

    return ( i + 1, current_activation )

def build_network(

):
    X = tf.placeholder()
    hidden_states = initialize hidden state to zeros
    i = tf.constant(0.)
    cond = lambda i: tf.less( i, num_layers )
    tf.while_loop(
        cond,
        build_feed_forward_layer,
        [ i, X, hidden_states ]
    )
    for t in time_steps:
        for i in layers:
            activations[i][t] += build_recurrent_term()

    return y

y = build_network()