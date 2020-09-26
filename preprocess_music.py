from midi_parser import MIDI_parser
import config_music as config
from utils import get_quant_time
import numpy as np
import argparse
import pathlib
import dload


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--midi_dir', type=str, default='maestro',
                            help='Directory where the midi files are stored')
    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_music',
                            help='Directory where the npz files will be stored')
    arg_parser.add_argument('-n', '--n_files', type=int, default=None,
                            help='Number of files to take into account (default: all)')
    arg_parser.add_argument('-d', '--download', action='store_true',
                            help='If activated the MAESTRO dataset will be downloaded (mandatory for the first time)')

    args = arg_parser.parse_args()

    if args.download:
        if not pathlib.Path(args.midi_dir).exists():
            pathlib.Path(args.midi_dir).mkdir(parents=True, exist_ok=True)
        else:
            assert pathlib.Path(args.midi_dir).is_dir()

    if pathlib.Path(args.npz_dir).exists():
        assert pathlib.Path(args.npz_dir).is_dir()
    else:
        pathlib.Path(args.npz_dir).mkdir(parents=True, exist_ok=True)

    if not args.n_files is None:
        assert isinstance(args.n_files, int)
        assert args.n_files > 0

    # ============================================================
    # ============================================================

    if args.download:

        pathlib.Path(args.midi_dir).mkdir(parents=True, exist_ok=True)
        print('Downloading dataset...')
        dload.save_unzip(config.dataset_url, args.midi_dir)

    ext_list = ['*.midi', '*.mid']

    midi_filenames = []
    for ext in ext_list:
        ext_filenames = pathlib.Path(args.midi_dir).rglob(ext)
        ext_filenames = list(map(lambda x: str(x), ext_filenames))
        midi_filenames += ext_filenames
    print(f'Found {len(midi_filenames)} midi files')
    assert len(midi_filenames) > 0

    if not args.n_files is None:
        n_files = max(0, min(args.n_files, len(midi_filenames)))
        midi_filenames = np.random.choice(
            midi_filenames, n_files, replace=False)
        assert len(midi_filenames) > 0

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

    print('Creating npz files...')
    midi_parser.preprocess_dataset(src_filenames=midi_filenames,
                                   dst_dir=args.npz_dir, batch_size=20, dst_filenames=None)

    print(f'Created dataset with {len(midi_filenames)} files')
