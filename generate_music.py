from midi_parser import MIDI_parser
from model import Music_transformer
import config_music as config
from utils import get_quant_time, generate_midis
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib
import matplotlib.pyplot as plt


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('n_songs', type=int,
                            help='Number of files to generate')

    arg_parser.add_argument('checkpoint_path', type=str,
                            help='Path to the saved weights')

    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_music',
                            help='Directory with the npz files')

    arg_parser.add_argument('-o', '--dst_dir', type=str, default='generated_midis',
                            help='Directory where the generated midi files will be stored')

    arg_parser.add_argument('-l', '--gen_len', type=int, default=6000,
                            help='Length of the generated midis (in midi messages)')

    arg_parser.add_argument('-k', '--top_k', type=int, default=3)

    arg_parser.add_argument('-t', '--temp', type=float, default=0.35,
                            help='Temperature of softmax')

    arg_parser.add_argument('-f', '--filenames', nargs='+', type=str, default=None,
                            help='Names of the generated midis. Length must be equal to n_songs')

    arg_parser.add_argument('-v', '--visualize_attention', action='store_true',
                            help='If activated, the attention weights will be saved as images')

    args = arg_parser.parse_args()

    assert isinstance(args.n_songs, int)
    assert args.n_songs > 0
    assert pathlib.Path(args.checkpoint_path).is_file()
    assert pathlib.Path(args.npz_dir).is_dir()
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

    npz_filenames = list(pathlib.Path(args.npz_dir).rglob('*.npz'))
    assert len(npz_filenames) > 0
    filenames_sample = np.random.choice(
        npz_filenames, args.n_songs, replace=False)

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
    model, _ = Music_transformer.build_from_config(
        config=config, checkpoint_path=args.checkpoint_path)

    midi_list, _, attention_weight_list, _ = generate_midis(model=model, seq_len=config.seq_len,
                                                            mem_len=config.mem_len, max_len=args.gen_len,
                                                            parser=midi_parser, filenames=filenames_sample,
                                                            pad_idx=config.pad_idx, top_k=args.top_k,
                                                            temp=args.temp)

    for midi, filename in zip(midi_list, midi_filenames):
        midi.save(filename)

    if args.visualize_attention:

        viz_dir = 'vizualized_attention'
        pathlib.Path(viz_dir).mkdir(parents=True, exist_ok=True)

        for layer_idx, layer_weights in enumerate(attention_weight_list, 1):
            for head_idx, head_weights in enumerate(layer_weights[0, ...].numpy(), 1):

                img_path = os.path.join(
                    viz_dir, f'layer{layer_idx}_head{head_idx}.png')
                plt.figure(figsize=(17, 14))
                plt.step(np.arange(head_weights.shape[1]), head_weights[0])
                #plt.imsave(img_path, head_weights, cmap='Reds')
                plt.savefig(img_path)
