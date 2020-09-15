from text_parser import Scam_parser
import config_text as config
import numpy as np
import argparse
import pathlib


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('corpus_path', type=str,
                            help='Path to the file that contains the corpus')

    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_text',
                            help='Directory where the npy files will be stored')

    arg_parser.add_argument('-t', '--tokenizer', type=str,
                            default=None, help='Path to a saved tokenizer')

    args = arg_parser.parse_args()

    assert pathlib.Path(args.corpus_path).is_file()
    if pathlib.Path(args.npy_dir).exists():
        assert pathlib.Path(args.npy_dir).is_dir()
    else:
        pathlib.Path(args.npy_dir).mkdir(parents=True, exist_ok=True)

    if not args.tokenizer is None:
        assert pathlib.Path(args.tokenizer).is_file()

    # ============================================================
    # ============================================================

    scam_parser = Scam_parser.build_from_config(config)
    scam_parser.preprocess_dataset(corpus_path=args.corpus_path,
                                   n_words=config.n_words,
                                   npy_dir=args.npy_dir,
                                   tokenizer=args.tokenizer)
