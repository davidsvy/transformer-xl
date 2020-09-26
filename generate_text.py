from text_parser import Scam_parser
from model import Gated_Transformer_XL
import config_text as config
from utils import softmax_with_temp, generate_text
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib
import tqdm
import pickle
import re


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('n_samples', type=int,
                            help='Number of samples to generate')

    arg_parser.add_argument('checkpoint_path', type=str,
                            help='Path to the saved weights')

    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_text',
                            help='Directory where the npy files are stored')

    arg_parser.add_argument('-o', '--dst_path', type=str, default='generated_text.txt',
                            help='Path where the generated text will be stored')

    arg_parser.add_argument('-l', '--gen_len', type=int, default=1000,
                            help='Maximum length of the generated text (in tokens)')

    arg_parser.add_argument('-k', '--top_k', type=int, default=3)

    arg_parser.add_argument('-t', '--temp', type=float, default=0.4,
                            help='Temperature of softmax')

    arg_parser.add_argument('-bl', '--beginning_list',
                            nargs='+', type=str, default=None)
    arg_parser.add_argument('-bs', '--beginning_string',
                            type=str, default=None)

    args = arg_parser.parse_args()

    assert isinstance(args.n_samples, int)
    assert args.n_samples > 0
    assert pathlib.Path(args.checkpoint_path).is_file()

    assert pathlib.Path(args.npy_dir).is_dir()
    tokenizer_path = os.path.join(args.npy_dir, 'tokenizer.pickle')
    stored_tokens_path = os.path.join(args.npy_dir, 'stored_tokens.pickle')
    assert pathlib.Path(tokenizer_path).is_file()
    assert pathlib.Path(stored_tokens_path).is_file()

    assert isinstance(args.gen_len, int)
    assert args.gen_len > 0
    assert isinstance(args.top_k, int)
    assert args.top_k > 0
    assert isinstance(args.temp, float)
    assert args.temp > 0.0
    if not args.beginning_list is None:
        assert isinstance(args.beginning_list, list)
        assert len(args.beginning_list) == args.n_samples
        for elem in args.beginning_list:
            assert isinstance(elem, str)
    if not args.beginning_string is None:
        assert isinstance(args.beginning_string, str)

    # ============================================================
    # ============================================================

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        assert isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer)
    with open(stored_tokens_path, 'rb') as handle:
        stored_tokens = pickle.load(handle)
        assert isinstance(stored_tokens, dict)

    scam_parser = Scam_parser.build_from_config(config)

    start_idx = tokenizer.word_index[config.start_token]
    end_idx = tokenizer.word_index[config.end_token]
    unknown_idx = tokenizer.word_index[config.unknown_token]
    blocked_idxs = [unknown_idx, config.pad_idx]

    if not args.beginning_list is None:
        beginning = args.beginning_list
    elif not args.beginning_string is None:
        beginning = args.beginning_string
    else:
        beginning = None

    model, _ = Gated_Transformer_XL.build_from_config(
        config=config, checkpoint_path=args.checkpoint_path)

    generated_features, _ = generate_text(model=model, seq_len=config.seq_len,
                                          mem_len=config.mem_len, max_len=args.gen_len,
                                          tokenizer=tokenizer, start_idx=start_idx,
                                          end_idx=end_idx, blocked_idxs=blocked_idxs,
                                          batch_size=args.n_samples, beginning=beginning,
                                          top_k=args.top_k, temp=args.temp)

    generated_texts = scam_parser.features_to_text(features=generated_features,
                                                   tokenizer=tokenizer,
                                                   stored_tokens=stored_tokens)

    delimiter = '\n' * 4 + ('#'*80 + '\n') * 4 + '\n' * 4
    generated_texts = delimiter.join(generated_texts)

    with open(args.dst_path, 'w', encoding='ISO-8859-1') as file:
        file.write(generated_texts)
