# -*- coding: utf-8 -*-

import itertools
import numpy as np
import random


time_steps = 3
batch_size = 4
num_input_nodes = 2
num_target_nodes = 1


def generate_data(
    max_num_digits,
    min_num_digits=0
):
    '''
        Returns samples in the shape (num_samples, time_steps, (x1,x2,y1)).
        XXX Should this yield one sample at a time? Sure why not.
    '''
    # num_digits = max_num_digits - 1
    num_digits = max_num_digits
    samples = []

    # Here I am deciding to take *all* the samples of length num_digits
    # (plus a couple of samples of length num_digits + 1)
    while True:
        for sequence in itertools.product([0, 1], repeat=2*num_digits):

            # Introduce some randomness
            # The more samples I filter out, the more computation needed
            # XXX Have it somehow w.r.t to total number of samples for this max_num_digits?
            if random.random() < 0.75:
                continue

            sample = []
            carry = 0

            for time_step in range(num_digits):

                x_1 = sequence[time_step]
                x_2 = sequence[num_digits + time_step]

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

            # This causes some samples to be longer 
            # than the specified num_digits
            # Can just be ignored while I want to keep sequence length constant
            # if carry:
            #     sample.append([0,0,1])

            # samples.append(sample)
            yield sample

        # XXX Instead of this I could kick out the -1 at the top and do
        # filtering here 
        # samples = np.array([x + [[0,0,0]] if len(x) == num_digits else x
        #                     for x in samples])
        # samples = np.array([x for x in samples if len(x) == num_digits])

        # return samples


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


batch_gen = generate_batches(generate_data, time_steps, batch_size)

X, y = next(batch_gen)

print(X.shape, y.shape)
print((X, y))



