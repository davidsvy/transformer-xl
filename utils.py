import tensorflow as tf
import numpy as np

__all__ = ('pad_ragged_2d', 'shuffle_ragged_2d',
           'inputs_to_labels', 'get_pos_encoding', 'get_quant_time')


def pad_ragged_2d(ragged_tensor, pad_idx):

    # ragged_tensor -> RAGGED(batch_size, None)
    lens = ragged_tensor.row_lengths(axis=-1)
    maxlen = tf.math.reduce_max(lens)
    mask = tf.sequence_mask(lens, maxlen, tf.bool)

    zero_padded = ragged_tensor.to_tensor()
    # zero_padded -> (batch_size, maxlen)

    padding = tf.constant(pad_idx, dtype=zero_padded.dtype)

    padded_tensor = tf.where(mask, zero_padded, padding)
    # padded_tensor -> (batch_size, maxlen)

    return padded_tensor


def shuffle_ragged_2d(ragged_tensor, pad_idx):

    # ragged_tensor -> RAGGED(batch_size, None)
    lens = ragged_tensor.row_lengths(axis=-1)
    second_lowest = -tf.nn.top_k(-lens, 2).values[-1]
    shuffled_slices = []

    for len_, row in zip(lens, ragged_tensor):
        if len_ <= second_lowest:
            new_row = tf.pad(
                row, paddings=[[0, second_lowest - len_]], constant_values=pad_idx)
        else:
            start_idx = tf.random.uniform(
                (), minval=0, maxval=len_ - second_lowest + 1, dtype=tf.int64)
            new_row = row[start_idx: start_idx + second_lowest]
        shuffled_slices.append(new_row[tf.newaxis, :])

    shuffled_tensor = tf.concat(shuffled_slices, axis=0)

    return shuffled_tensor


def inputs_to_labels(inputs, pad_idx):

    # inputs -> (batch_size, seq_len)
    inputs_padded = tf.pad(inputs[:, 1:], paddings=[
                           [0, 0], [0, 1]], constant_values=pad_idx)

    return inputs_padded


def get_pos_encoding(seq_len, d_model):

    numerator = np.arange(seq_len, dtype=np.float32)
    numerator = numerator[:, np.newaxis]

    denominator = np.arange(0, d_model, 2, dtype=np.float32)
    denominator = denominator / d_model

    denominator = np.power(np.array(10000, dtype=np.float32), denominator)

    denominator = 1 / denominator
    denominator = np.repeat(denominator, 2)
    denominator = denominator[np.newaxis, :]

    encoding = np.matmul(numerator, denominator)
    encoding[:, ::2] = np.sin(encoding[:, ::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    #encoding = encoding[np.newaxis, ...]
    encoding = tf.cast(encoding, dtype=tf.float32)

    return encoding


def get_quant_time():

    step = 0.001
    coef = 1.1
    delta = 0
    total_reps = 120
    local_reps = 3
    quant_time = []
    for _ in range(total_reps // local_reps):
        for _ in range(local_reps):
            delta += step
            quant_time.append(delta)

        step *= coef

    quant_time = np.sort(quant_time + [5.0, 0.0])
    return quant_time


def softmax_with_temp(x, temp=1.0):

    assert isinstance(temp, float)
    assert temp > 0
    assert all(map(lambda a: a > 0, x))

    x = x / np.sum(x) / temp
    x = tf.nn.softmax(x).numpy()
    return x
