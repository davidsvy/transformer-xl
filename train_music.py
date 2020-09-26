from midi_parser import MIDI_parser
from model import Music_transformer
import config_music as config
from utils import shuffle_ragged_2d, inputs_to_labels, get_quant_time
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_music',
                            help='Directory where the npz files are stored')

    arg_parser.add_argument('-c', '--checkpoint_dir', type=str, default='checkpoints_music',
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

    assert pathlib.Path(args.npz_dir).is_dir()
    if pathlib.Path(args.checkpoint_dir).exists():
        assert pathlib.Path(args.checkpoint_dir).is_dir()
    else:
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    assert isinstance(args.checkpoint_period, int)
    assert args.checkpoint_period > 0
    if not args.weights is None:
        assert pathlib.Path(args.weights).is_file()
        assert not args.optimizer is None
        assert pathlib.Path(args.optimizer).is_file()

    # ============================================================
    # ============================================================

    tf.config.experimental_run_functions_eagerly(False)

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

    print('Creating dataset')
    dataset = midi_parser.get_tf_dataset(
        file_directory=args.npz_dir, batch_size=config.batch_size,
        n_samples=args.n_files)

    batches_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
    assert batches_per_epoch > 0
    print(f'Created dataset with {batches_per_epoch} batches per epoch')

    model, optimizer = Music_transformer.build_from_config(config=config, checkpoint_path=args.weights,
                                                           optimizer_path=args.optimizer)

    loss_metric = tf.keras.metrics.Mean(name='loss')
    acc_metric_sound = tf.keras.metrics.SparseCategoricalAccuracy(
        name='acc_sound')
    acc_metric_delta = tf.keras.metrics.SparseCategoricalAccuracy(
        name='acc_delta')

    use_attn_reg = config.use_attn_reg

    @tf.function
    def train_step(inputs_sound, inputs_delta, labels_sound, labels_delta, mem_list):

        with tf.GradientTape() as tape:

            logits_sound, logits_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
                inputs=(inputs_sound, inputs_delta),
                mem_list=mem_list,
                next_mem_len=mem_len,
                training=True
            )

            if use_attn_reg:
                attention_loss = 4 * tf.math.reduce_mean(attention_loss_list)
            else:
                attention_loss = None

            loss, pad_mask = model.get_loss(
                logits_sound=logits_sound,
                logits_delta=logits_delta,
                labels_sound=labels_sound,
                labels_delta=labels_delta,
                attention_loss=attention_loss
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        outputs_sound = tf.nn.softmax(logits_sound, axis=-1)
        # outputs_sound -> (batch_size, seq_len, n_sounds)
        outputs_delta = tf.nn.softmax(logits_delta, axis=-1)
        # outputs_delta -> (batch_size, seq_len, n_deltas)

        non_padded_labels_sound = tf.boolean_mask(labels_sound, pad_mask)
        non_padded_outputs_sound = tf.boolean_mask(outputs_sound, pad_mask)

        non_padded_labels_delta = tf.boolean_mask(labels_delta, pad_mask)
        non_padded_outputs_delta = tf.boolean_mask(outputs_delta, pad_mask)

        loss_metric(loss)
        acc_metric_sound(non_padded_labels_sound, non_padded_outputs_sound)
        acc_metric_delta(non_padded_labels_delta, non_padded_outputs_delta)

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

    for epoch in range(1, n_epochs + 1):

        print(f"\nEpoch {epoch}/{n_epochs}")

        progress_bar = tf.keras.utils.Progbar(batches_per_epoch, stateful_metrics=[
            'acc_sound', 'acc_delta', 'loss'])

        loss_metric.reset_states()
        acc_metric_sound.reset_states()
        acc_metric_delta.reset_states()

        for batch_ragged in dataset:

            batch_sound, batch_delta = shuffle_ragged_2d(batch_ragged, pad_idx)
            # batch_sound -> (batch_size, maxlen)
            # batch_delta -> (batch_size, maxlen)

            batch_labels_sound = inputs_to_labels(batch_sound, pad_idx)
            # batch_labels_sound -> (batch_size, maxlen)
            batch_labels_delta = inputs_to_labels(batch_delta, pad_idx)
            # batch_labels_delta -> (batch_size, maxlen)

            maxlen = batch_sound.shape[1]
            if maxlen < seq_len + 100:
                continue

            # ======================================================================================
            # train on random slices of the batch
            # ======================================================================================

            segs_per_batch = min(max_segs_per_batch, maxlen // seq_len)
            mem_list = None
            start = np.random.randint(
                0, maxlen - (segs_per_batch) * seq_len + 1)

            for _ in range(segs_per_batch):

                seg_sound = batch_sound[:, start: start + seq_len]
                # seg_sound -> (batch_size, seq_len)
                seg_delta = batch_delta[:, start: start + seq_len]
                # seg_delta -> (batch_size, seq_len)

                seg_labels_sound = batch_labels_sound[:,
                                                      start: start + seq_len]
                # seg_labels_sound -> (batch_size, seq_len)
                seg_labels_delta = batch_labels_delta[:,
                                                      start: start + seq_len]
                # seg_labels_delta -> (batch_size, seq_len)

                # ============================
                # training takes place here
                # ============================
                mem_list = train_step(inputs_sound=seg_sound,
                                      inputs_delta=seg_delta,
                                      labels_sound=seg_labels_sound,
                                      labels_delta=seg_labels_delta,
                                      mem_list=mem_list)

                start += seq_len

            # training for this batch is over

            values = [('acc_sound', acc_metric_sound.result()),
                      ('acc_delta', acc_metric_delta.result()),
                      ('loss', loss_metric.result())]

            progress_bar.add(1, values=values)

        if epoch % args.checkpoint_period == 0:

            checkpoint_path = os.path.join(
                args.checkpoint_dir, f'checkpoint{epoch}.h5')
            model.save_weights(checkpoint_path)

            optimizer_path = os.path.join(
                args.checkpoint_dir, f'optimizer{epoch}.npy')
            np.save(optimizer_path, optimizer.get_weights())

            print(f'Saved model weights at {checkpoint_path}')
            print(f'Saved optimizer weights at {optimizer_path}')
