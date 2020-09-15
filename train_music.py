from midi_parser import MIDI_parser
from model import Transformer_XL
import config_music as config
from utils import shuffle_ragged_2d, inputs_to_labels, get_quant_time
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_music',
                            help='Directory where the npy files are stored')

    arg_parser.add_argument('-c', '--checkpoint_dir', type=str, default='checkpoints_music',
                            help='Directory where the saved weights will be stored')

    arg_parser.add_argument('-p', '--checkpoint_period', type=int, default=1,
                            help='Number of epochs between saved checkpoints')

    arg_parser.add_argument('-n', '--n_files', type=int, default=None,
                            help='Number of dataset files to take into account (default: all)')

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

    idx_to_time = get_quant_time()

    parser = MIDI_parser(
        tempo=config.tempo, ppq=config.ppq, numerator=config.numerator,
        denominator=config.denominator, clocks_per_click=config.clocks_per_click,
        notated_32nd_notes_per_beat=config.notated_32nd_notes_per_beat,
        cc_kept=config.cc_kept, cc_threshold=config.cc_threshold, cc_lower=config.cc_lower,
        cc_upper=config.cc_upper, n_notes=config.n_notes, n_times=config.n_times,
        vel_value=config.vel_value, idx_to_time=idx_to_time, n_jobs=config.n_jobs)

    print('Creating dataset')
    dataset = parser.get_tf_dataset(
        file_directory=args.npy_dir, batch_size=config.batch_size,
        buffer_size=config.buffer_size, n_samples=args.n_files)

    batches_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
    assert batches_per_epoch > 0
    print(f'Created dataset with {batches_per_epoch} batches per epoch')

    model = Transformer_XL.build_from_config(config, args.weights)

    optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    loss_metric = tf.keras.metrics.Mean(name='loss')
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

    @tf.function
    def train_step(inputs, labels, mem_inputs):

        with tf.GradientTape() as tape:

            trash, mem_list = model(inputs=mem_inputs,
                                    mem_list=None,
                                    next_mem_len=None,
                                    training=True)

            logits, trash = model(inputs=inputs,
                                  mem_list=mem_list,
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

        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")

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

            # ======================================================================================
            # train on random slices of the batch
            # ======================================================================================

            segs_per_batch = min(max_segs_per_batch,
                                 maxlen // (seq_len + mem_len))

            for _ in range(segs_per_batch):

                start = tf.random.uniform(
                    shape=(), minval=0, maxval=maxlen - (seq_len + mem_len) - 1, dtype=tf.int32)

                seg_mem = batch_inputs[:, start: start + mem_len]
                # seg_mem -> (batch_size, mem_len)

                seg_inputs = batch_inputs[:, start +
                                          mem_len: start + mem_len + seq_len]
                # seg_inputs -> (batch_size, seq_len)
                seg_labels = batch_labels[:, start +
                                          mem_len: start + mem_len + seq_len]
                # seg_labels -> (batch_size, seq_len)

                # ============================
                # training takes place here
                # ============================
                train_step(seg_inputs, seg_labels, seg_mem)

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
