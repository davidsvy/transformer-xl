from text_parser import Scam_parser
from model import Transformer_XL
import config_text as config
from utils import softmax_with_temp
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib
import tqdm
import pickle
import re


def generate_text(model, seq_len, mem_len, max_len, tokenizer, start_idx, end_idx, blocked_idxs,
                  batch_size, beginning=None, top_k=1, temp=1.0):

    if isinstance(beginning, str):
        generated_idxs = tokenizer.texts_to_sequences([beginning])
        generated_idxs = np.repeat(generated_idxs, batch_size, axis=0)
        start_idxs = np.full((batch_size, 1), start_idx,
                             dtype=generated_idxs.dtype)
        generated_idxs = np.concatenate((start_idxs, generated_idxs), axis=-1)

    elif isinstance(beginning, list):
        assert len(beginning) == batch_size
        for string in beginning:
            assert isinstance(string, str)
        generated_idxs = tokenizer.texts_to_sequences(beginning)
        min_len = min([len(x) for x in generated_idxs])
        generated_idxs = np.array([x[:min_len] for x in generated_idxs])
        start_idxs = np.full((batch_size, 1), start_idx,
                             dtype=generated_idxs.dtype)
        generated_idxs = np.concatenate((start_idxs, generated_idxs), axis=-1)

    else:
        generated_idxs = np.full((batch_size, 1), start_idx)

    end_flags = [False] * batch_size
    end_cnt = 0

    orig_len = generated_idxs.shape[1]
    assert orig_len >= 1
    # generated_idxs -> (batch_size, orig_len)

    for _ in tqdm.tqdm(range(max_len - orig_len)):

        if generated_idxs.shape[1] <= seq_len:

            inputs = tf.constant(generated_idxs, dtype=tf.int32)
            # inputs -> (batch_size, cur_length)

            outputs, _ = model(inputs=inputs,
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

            # outputs -> (batch_size, cur_length, num_words)
            outputs, _ = model(inputs=inputs,
                               mem_list=mem_list,
                               next_mem_len=None,
                               training=False)

        outputs = tf.nn.softmax(outputs, axis=-1)

        probs = outputs[:, -1, :].numpy()
        # probs -> (batch_size, num_words)
        if not blocked_idxs is None:
            probs[:, blocked_idxs] = 0

        generated_new = []
        for batch_idx, batch_probs in enumerate(probs):

            best_idxs = batch_probs.argsort()[-top_k:][::-1]
            best_probs = softmax_with_temp(batch_probs[best_idxs], temp)
            new_idx = np.random.choice(best_idxs, p=best_probs)
            generated_new.append(new_idx)
            if new_idx == end_idx and not end_flags[batch_idx]:
                end_flags[batch_idx] = True
                end_cnt += 1

        generated_new = np.array(generated_new)[:, np.newaxis]
        # generated_new -> (batch_size, 1)
        generated_idxs = np.concatenate(
            (generated_idxs, generated_new), axis=-1)
        if end_cnt >= batch_size:
            break

    return generated_idxs


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
                            help='Maximum length of the generated text (in words)')

    arg_parser.add_argument('-k', '--top_k', type=int, default=6)

    arg_parser.add_argument('-t', '--temp', type=float, default=0.5,
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

    model = Transformer_XL.build_from_config(
        config=config, checkpoint_path=args.checkpoint_path)

    generated_features = generate_text(model=model, seq_len=config.seq_len,
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
