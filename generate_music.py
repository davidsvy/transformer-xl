from midi_parser import MIDI_parser
from model import Transformer_XL
import config_music as config
from utils import get_quant_time, softmax_with_temp
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib
import tqdm


def generate_midis(model, seq_len, mem_len, max_len, parser, filenames, pad_idx, top_k=1, temp=1.0):

    assert isinstance(seq_len, int)
    assert seq_len > 0

    assert isinstance(mem_len, int)
    assert mem_len >= 0

    assert isinstance(max_len, int)
    assert max_len > 1

    batch_size = len(filenames)

    feature_list = [np.load(file) for file in filenames]
    min_len = min([len(f) for f in feature_list])

    orig_len = np.random.randint(1, min(mem_len, min_len, max_len))
    assert orig_len >= 1

    generated_idxs = np.array([arr[:orig_len] for arr in feature_list])
    # generated_idxs -> (batch_size, orig_len)

    delta_first, delta_last = parser.time_range
    sound_first, sound_last = parser.sound_range

    delta_flag = False
    sound_flag = not delta_flag

    states = [delta_flag] * batch_size

    for batch_idx, feature in enumerate(generated_idxs[:, -1]):
        if delta_first <= feature < delta_last:
            states[batch_idx] = delta_flag
        elif sound_first <= feature < sound_last:
            states[batch_idx] = sound_flag
        else:
            raise Exception(f'Invalid feature: {feature}')

    for _ in tqdm.tqdm(range(max_len - orig_len)):

        if generated_idxs.shape[1] <= seq_len:

            # input -> (1, cur_length)
            inputs = tf.constant(generated_idxs, dtype=tf.int32)

            outputs, mem_list = model(inputs=inputs,
                                      mem_list=None,
                                      next_mem_len=None,
                                      training=False)

        else:

            cur_mem_len = min(mem_len, generated_idxs.shape[1] - seq_len)
            memory = generated_idxs[:, -(cur_mem_len + seq_len):-seq_len]
            memory = tf.constant(memory, dtype=tf.int32)
            # memory -> (batch_size, cur_mem_len)

            _, mem_list = model(inputs=memory,
                                mem_list=None,
                                next_mem_len=mem_len,
                                training=False)

            # insert input
            inputs = generated_idxs[:, -seq_len:]
            inputs = tf.constant(inputs, dtype=tf.int32)

            # output -> (batch_size, cur_length, num_words)
            outputs, _ = model(inputs=inputs,
                               mem_list=mem_list,
                               next_mem_len=None,
                               training=False)

        outputs = tf.nn.softmax(outputs, axis=-1)

        probs = outputs[:, -1, :].numpy()
        # probs -> (batch_size, num_words)
        probs[:, pad_idx] = 0

        generated_new = []

        for batch_idx, batch_probs in enumerate(probs):

            batch_probs[delta_first: delta_first + 12] = 0

            if states[batch_idx] == delta_flag:
                batch_probs[delta_first: delta_last] = 0
                states[batch_idx] = sound_flag
            elif states[batch_idx] == sound_flag:
                batch_probs[sound_first: sound_last] = 0
                states[batch_idx] = delta_flag
            else:
                raise Exception(f'Invalid flag: {states[batch_idx]}')

            best_idxs = batch_probs.argsort()[-top_k:][::-1]
            best_probs = softmax_with_temp(batch_probs[best_idxs], temp)
            new_idx = np.random.choice(best_idxs, p=best_probs)
            generated_new.append(new_idx)

        generated_new = np.array(generated_new)[:, np.newaxis]
        # generated_new -> (batch_size, 1)
        generated_idxs = np.concatenate(
            (generated_idxs, generated_new), axis=-1)

    midi_list = [parser.features_to_midi(features)
                 for features in generated_idxs]
    return midi_list, generated_idxs, orig_len


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('n_songs', type=int,
                            help='Number of files to generate')

    arg_parser.add_argument('checkpoint_path', type=str,
                            help='Path to the saved weights')

    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_music',
                            help='Directory with the npy files')

    arg_parser.add_argument('-o', '--dst_dir', type=str, default='generated_midis',
                            help='Directory where the generated midi files will be stored')

    arg_parser.add_argument('-l', '--gen_len', type=int, default=6000,
                            help='Length of the generated midis (in features)')

    arg_parser.add_argument('-k', '--top_k', type=int, default=6)

    arg_parser.add_argument('-t', '--temp', type=float, default=0.3,
                            help='Temperature of softmax')

    arg_parser.add_argument('-f', '--filenames', nargs='+', type=str, default=None,
                            help='Names of the generated midis. Length mus be equal to n_songs')

    args = arg_parser.parse_args()

    assert isinstance(args.n_songs, int)
    assert args.n_songs > 0
    assert pathlib.Path(args.checkpoint_path).is_file()
    assert pathlib.Path(args.npy_dir).is_dir()
    if pathlib.Path(args.dst_dir).exists():
        assert pathlib.Path(args.dst_dir).is_dir()
    else:
        pathlib.Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
    assert isinstance(args.gen_len, int)
    assert args.gen_len > 0
    assert isinstance(args.top_k, int)
    assert args.top_k > 0
    assert isinstance(args.temp, float)
    assert args.temp > 0.0
    if args.filenames is None:
        midi_filenames = [str(i) for i in range(1, args.n_songs + 1)]
    else:
        midi_filenames = args.filenames
    midi_filenames = [f + '.midi' for f in midi_filenames]
    midi_filenames = [os.path.join(args.dst_dir, f) for f in midi_filenames]
    assert len(midi_filenames) == args.n_songs
    assert len(set(midi_filenames)) == len(midi_filenames)

    # ============================================================
    # ============================================================

    npy_filenames = list(pathlib.Path(args.npy_dir).rglob('*.npy'))
    assert len(npy_filenames) > 0
    filenames_sample = np.random.choice(
        npy_filenames, args.n_songs, replace=False)

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser(
        tempo=config.tempo, ppq=config.ppq, numerator=config.numerator,
        denominator=config.denominator, clocks_per_click=config.clocks_per_click,
        notated_32nd_notes_per_beat=config.notated_32nd_notes_per_beat,
        cc_kept=config.cc_kept, cc_threshold=config.cc_threshold, cc_lower=config.cc_lower,
        cc_upper=config.cc_upper, n_notes=config.n_notes, n_times=config.n_times,
        vel_value=config.vel_value, idx_to_time=idx_to_time, n_jobs=config.n_jobs)

    model = Transformer_XL.build_from_config(
        config=config, checkpoint_path=args.checkpoint_path)

    midi_list, _, _ = generate_midis(model=model,
                                     seq_len=config.seq_len,
                                     mem_len=config.mem_len,
                                     max_len=args.gen_len,
                                     parser=midi_parser,
                                     filenames=filenames_sample,
                                     pad_idx=config.pad_idx,
                                     top_k=args.top_k,
                                     temp=args.temp)

    for midi, filename in zip(midi_list, midi_filenames):
        midi.save(filename)
