from text_parser import Scam_parser
from model import Gated_Transformer_XL
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

    arg_parser.add_argument('-n', '--n_files', type=int, default=None,
                            help='Number of dataset files to take into account (default: all)')

    arg_parser.add_argument('-w', '--weights', type=str,
                            default=None, help='Path to saved model weights')

    arg_parser.add_argument('-o', '--optimizer', type=str,
                            default=None, help='Path to saved optimizer weights')

    args = arg_parser.parse_args()

    assert pathlib.Path(args.npy_dir).is_dir()
    if pathlib.Path(args.checkpoint_dir).exists():
        assert pathlib.Path(args.checkpoint_dir).is_dir()
    else:
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    assert isinstance(args.checkpoint_period, int)
    assert args.checkpoint_period > 0
    assert isinstance(args.n_files, int)
    assert args.n_files > 0

    if not args.weights is None:
        assert pathlib.Path(args.weights).is_file()
        assert not args.optimizer is None
        assert pathlib.Path(args.optimizer).is_file()

    # ============================================================
    # ============================================================

    tf.config.experimental_run_functions_eagerly(False)

    scam_parser = Scam_parser.build_from_config(config)

    print('Loading dataset...')
    dataset = scam_parser.get_tf_dataset(file_directory=args.npy_dir,
                                         batch_size=config.batch_size,
                                         n_samples=args.n_files)

    batches_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
    assert batches_per_epoch > 0
    print(f'Loaded dataset with {batches_per_epoch} batches per epoch')

    loss_metric = tf.keras.metrics.Mean(name='loss')
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

    model, optimizer = Gated_Transformer_XL.build_from_config(
        config, args.weights)

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
    def train_step(inputs, labels, mem_list):

        with tf.GradientTape() as tape:

            logits, next_mem_list, attention_weight_list, attention_loss_list = model(
                inputs=inputs,
                mem_list=mem_list,
                next_mem_len=mem_len,
                training=True
            )

            attention_loss = 4 * tf.math.reduce_mean(attention_loss_list)

            loss, pad_mask = model.get_loss(
                logits=logits,
                labels=labels,
                attention_loss=attention_loss
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        outputs = tf.nn.softmax(logits, axis=-1)
        # outputs -> (batch_size, seq_len, n_classes)

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

    # =======================================

    for epoch in range(1, n_epochs + 1):

        print(f"\nEpoch {epoch}/{n_epochs}")

        progress_bar = tf.keras.utils.Progbar(
            batches_per_epoch, stateful_metrics=['acc', 'loss'])
        n_skipped = 0
        loss_metric.reset_states()
        acc_metric.reset_states()

        for batch_ragged in dataset:

            batch = shuffle_ragged_2d(batch_ragged, pad_idx, 2)[0]
            # batch -> (batch_size, max_len)

            batch_labels = inputs_to_labels(batch, pad_idx)
            # batch_labels -> (batch_size, max_len)

            max_len = batch.shape[1]
            if max_len < seq_len + 10:
                n_skipped += 1
                continue

            # ======================================================================================
            # train on random slices of the batch
            # ======================================================================================

            segs_per_batch = min(max_segs_per_batch, max_len // seq_len)
            mem_list = None
            start = 0

            for _ in range(segs_per_batch):

                seg = batch[:, start: start + seq_len]
                # seg -> (batch_size, seq_len)

                seg_labels = batch_labels[:, start: start + seq_len]
                # seg_labels -> (batch_size, seq_len)

                # ============================
                # training takes place here
                # ============================
                mem_list = train_step(inputs=seg,
                                      labels=seg_labels,
                                      mem_list=mem_list)

                start += seq_len

            # training for this batch is over

            values = [('acc', acc_metric.result()),
                      ('loss', loss_metric.result())]
            progress_bar.add(1, values=values)

        print(f'\nSkipped {n_skipped} segments')

        if epoch % args.checkpoint_period == 0:

            checkpoint_path = os.path.join(
                args.checkpoint_dir, f'checkpoint{epoch}.h5')
            model.save_weights(checkpoint_path)

            optimizer_path = os.path.join(
                args.checkpoint_dir, f'optimizer{epoch}.npy')
            np.save(optimizer_path, optimizer.get_weights())

            print(checkpoint_path)
            print(optimizer_path)

    # ======================================
