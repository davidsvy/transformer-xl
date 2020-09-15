from text_parser import Scam_parser
from model import Transformer_XL
import config_text as config
from utils import shuffle_ragged_2d, inputs_to_labels
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_text',
                            help='Directory where the npy files are stored')

    arg_parser.add_argument('-ch', '--checkpoint_dir', type=str, default='checkpoints_text',
                            help='Directory where the saved weights will be stored')

    arg_parser.add_argument('-p', '--checkpoint_period', type=int, default=1,
                            help='Number of epochs between saved checkpoints')

    arg_parser.add_argument('-w', '--weights', type=str,
                            default=None, help='Path to a saved checkpoint file')

    args = arg_parser.parse_args()

    assert pathlib.Path(args.npy_dir).is_dir()
    if pathlib.Path(args.checkpoint_dir).exists():
        assert pathlib.Path(args.checkpoint_dir).is_dir()
    else:
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    assert isinstance(args.checkpoint_period, int)
    assert args.checkpoint_period > 0

    if not args.weights is None:
        assert pathlib.Path(args.weights).is_file()

    # ============================================================
    # ============================================================

    scam_parser = Scam_parser.build_from_config(config)

    print('Loading dataset...')
    dataset = scam_parser.get_tf_dataset(file_directory=args.npy_dir,
                                         batch_size=config.batch_size,
                                         buffer_size=config.buffer_size)

    batches_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
    assert batches_per_epoch > 0
    print(f'Loaded dataset with {batches_per_epoch} batches per epoch')

    optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    loss_metric = tf.keras.metrics.Mean(name='loss')
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

    model = Transformer_XL.build_from_config(config, args.weights)

    @tf.function
    def first_train_step(inputs, labels):

        with tf.GradientTape() as tape:

            logits, mem_list = model(inputs=inputs,
                                     mem_list=None,
                                     next_mem_len=None,
                                     training=True)

            loss, pad_mask = model.get_loss(logits=logits, labels=labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        outputs = tf.nn.softmax(logits, axis=-1)
        # outputs -> (batch_size, seq_len, num_classes)

        non_padded_labels = tf.boolean_mask(labels, pad_mask)
        non_padded_outputs = tf.boolean_mask(outputs, pad_mask)

        loss_metric(loss)
        acc_metric(non_padded_labels, non_padded_outputs)

        return mem_list

    @tf.function
    def train_step(inputs, labels, mem_list, next_mem_len):

        with tf.GradientTape() as tape:

            logits, next_mem_list = model(inputs=inputs,
                                          mem_list=mem_list,
                                          next_mem_len=next_mem_len,
                                          training=True)

            loss, pad_mask = model.get_loss(logits=logits, labels=labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        outputs = tf.nn.softmax(logits, axis=-1)
        # outputs -> (batch_size, seq_len, num_classes)

        non_padded_labels = tf.boolean_mask(labels, pad_mask)
        non_padded_outputs = tf.boolean_mask(outputs, pad_mask)

        loss_metric(loss)
        acc_metric(non_padded_labels, non_padded_outputs)

        return next_mem_list

    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # ==============================   TRAINING LOOP   ====================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================

    n_epochs = config.n_epochs
    pad_idx = config.pad_idx
    seq_len = config.seq_len
    mem_len = config.mem_len
    max_segs_per_batch = config.max_segs_per_batch

    tf.config.experimental_run_functions_eagerly(False)

    for epoch in range(n_epochs):

        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        progress_bar = tf.keras.utils.Progbar(
            batches_per_epoch, stateful_metrics=['acc', 'loss'])

        acc_metric.reset_states()
        loss_metric.reset_states()

        for batch_ragged in dataset:

            batch_inputs = shuffle_ragged_2d(
                ragged_tensor=batch_ragged, pad_idx=pad_idx)
            # batch_inputs -> (batch_size, maxlen)

            batch_labels = inputs_to_labels(
                inputs=batch_inputs, pad_idx=pad_idx)
            # dur_labels -> (batch_size, maxlen)

            maxlen = batch_inputs.shape[1]
            if maxlen == 0:
                continue

            start = 0
            segs_per_batch = min(maxlen // seq_len, max_segs_per_batch)

            seg_inputs = batch_inputs[:, start: start + min(seq_len, maxlen)]
            # seg_inputs -> (batch_size, seq_len)
            seg_labels = batch_labels[:, start: start + min(seq_len, maxlen)]
            # seg_labels -> (batch_size, seq_len)

            mem_list = first_train_step(inputs=seg_inputs, labels=seg_labels)

            for _ in range(segs_per_batch - 1):

                start += seq_len

                seg_inputs = batch_inputs[:, start: start + seq_len]
                # seg_inputs -> (batch_size, seq_len)
                seg_labels = batch_labels[:, start: start + seq_len]
                # seg_labels -> (batch_size, seq_len)

                mem_list = train_step(inputs=seg_inputs, labels=seg_labels,
                                      mem_list=mem_list, next_mem_len=mem_len)

            # training for this batch is over

            values = [('acc', acc_metric.result()),
                      ('loss', loss_metric.result())]
            progress_bar.add(1, values=values)

        # training for this epoch is over

        if (epoch + 1) % args.checkpoint_period == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f'checkpoint{epoch + 1}.h5')
            model.save_weights(checkpoint_path)
            print(f'Saved checkpoint at {checkpoint_path}')
