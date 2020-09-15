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
    arg_parser.add_argument('-np', '--npy_dir', type=str, default='npy_music',
                            help='Directory where the npy files will be stored')
    arg_parser.add_argument('-n', '--n_files', type=int, default=-1,
                            help='Number of files to take into account (default: all)')
    arg_parser.add_argument('-d', '--download', action='store_true',
                            help='If activated the MAESTRO dataset will be downloaded (mandatory for the first time)')

    args = arg_parser.parse_args()

    if args.download:
        if not pathlib.Path(args.midi_dir).exists():
            pathlib.Path(args.midi_dir).mkdir(parents=True, exist_ok=True)
        else:
            assert pathlib.Path(args.midi_dir).is_dir()

    if pathlib.Path(args.npy_dir).exists():
        assert pathlib.Path(args.npy_dir).is_dir()
    else:
        pathlib.Path(args.npy_dir).mkdir(parents=True, exist_ok=True)

    assert isinstance(args.n_files, int)
    # ============================================================
    # ============================================================

    if args.download:

        pathlib.Path(args.midi_dir).mkdir(parents=True, exist_ok=True)
        print('Downloading dataset...')
        dload.save_unzip(config.dataset_url, args.midi_dir)

    midi_filenames = list(pathlib.Path(args.midi_dir).rglob('*.midi'))
    midi_filenames = list(map(lambda x: str(x), midi_filenames))
    assert len(midi_filenames) > 0

    if args.n_files > 0:
        n_files = max(0, min(args.n_files, len(midi_filenames)))
        midi_filenames = np.random.choice(
            midi_filenames, n_files, replace=False)

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser(
        tempo=config.tempo, ppq=config.ppq, numerator=config.numerator,
        denominator=config.denominator, clocks_per_click=config.clocks_per_click,
        notated_32nd_notes_per_beat=config.notated_32nd_notes_per_beat,
        cc_kept=config.cc_kept, cc_threshold=config.cc_threshold, cc_lower=config.cc_lower,
        cc_upper=config.cc_upper, n_notes=config.n_notes, n_times=config.n_times,
        vel_value=config.vel_value, idx_to_time=idx_to_time, n_jobs=config.n_jobs)

    print('Creating npy files...')
    midi_parser.preprocess_dataset(
        src_filenames=midi_filenames, dst_dir=args.npy_dir, batch_size=20)
    print(f'Created dataset with {len(midi_filenames)} files')
