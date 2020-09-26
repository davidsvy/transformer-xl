import tensorflow as tf
import numpy as np
import mido
import re
import os
import joblib
import glob
import tqdm
import pathlib
from collections import Counter

__all__ = ('MIDI_parser')


class MIDI_parser():

    def __init__(self, tempo, ppq, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat,
                 cc_kept, cc_threshold, cc_lower, cc_upper, n_notes, n_deltas, vel_value, idx_to_time, n_jobs):

        self.tempo = tempo
        self.ppq = ppq
        self.numerator = numerator
        self.denominator = denominator
        self.clocks_per_click = clocks_per_click
        self.notated_32nd_notes_per_beat = notated_32nd_notes_per_beat

        self.cc_kept = cc_kept
        self.cc_threshold = cc_threshold
        self.cc_lower = cc_lower
        self.cc_upper = cc_upper

        self.vel_value = vel_value

        assert n_notes <= 128
        assert 128 % n_notes == 0
        self.note_ratio = 128 // n_notes
        self.n_notes = n_notes

        self.n_cc = 2 * len(self.cc_kept)

        self.n_sounds = 2 * self.n_notes + self.n_cc + 1

        self.n_deltas = n_deltas

        self.pad_idx = 0
        self.n_jobs = n_jobs

        assert self.n_deltas - 1 == len(idx_to_time)
        assert idx_to_time[0] == 0
        assert np.sum(idx_to_time == 0) == 1

        self.idx_to_time = idx_to_time
        self.closest_neighbors = [
            (a + b) / 2 for a, b in zip(idx_to_time[1:-1], idx_to_time[2:])]

        self.note_on_offset = 1
        self.note_off_offset = self.note_on_offset + self.n_notes
        self.cc_offset = self.note_off_offset + self.n_notes

    def secs_to_ticks(self, secs):

        return int(round(1e6 * self.ppq / self.tempo * secs))

    def save_features(self, features, filename):

        sounds, deltas = features

        np.savez(filename, sounds=sounds, deltas=deltas)

    def load_features(self, filename):

        container = np.load(filename)
        sounds = container['sounds']
        deltas = container['deltas']

        return sounds, deltas

    def midi_to_features(self, src_file):

        midi = mido.MidiFile(src_file)
        sounds = []
        deltas = []

        for msg in midi:

            if msg.time == 0:
                time = 1
            else:
                time = 2 + np.digitize(msg.time, self.closest_neighbors)

            # note on
            if msg.type == 'note_on' and msg.velocity > 0:

                note_on = msg.note
                note_on = note_on // self.note_ratio
                note_on += self.note_on_offset

                sounds.append(note_on)
                deltas.append(time)

            # note_off
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):

                note_off = msg.note
                note_off = note_off // self.note_ratio
                note_off += self.note_off_offset

                sounds.append(note_off)
                deltas.append(time)

            # control_change
            elif msg.type == 'control_change' and msg.control in self.cc_kept:

                control_idx = self.cc_kept.index(msg.control)
                value = msg.value >= self.cc_threshold
                cc = control_idx * 2 + value
                cc += self.cc_offset

                sounds.append(cc)
                deltas.append(time)

        assert len(sounds) == len(deltas)
        sounds = np.array(sounds).astype(np.uint16)
        deltas = np.array(deltas).astype(np.uint8)

        return (sounds, deltas)

    def features_to_midi(self, sounds, deltas):

        assert len(sounds) == len(deltas)

        track = mido.MidiTrack()

        tempo = mido.MetaMessage('set_tempo', tempo=self.tempo, time=0)
        time_signature = mido.MetaMessage('time_signature', numerator=self.numerator, denominator=self.denominator,
                                          clocks_per_click=self.clocks_per_click,
                                          notated_32nd_notes_per_beat=self.notated_32nd_notes_per_beat, time=0)

        track.append(tempo)
        track.append(time_signature)

        mask = sounds != self.pad_idx
        sounds = sounds[mask]
        deltas = deltas[mask]

        for sound, delta in zip(sounds, deltas):

            delta_idx = delta - 1
            secs = self.idx_to_time[delta_idx]
            ticks = self.secs_to_ticks(secs)

            # note_on
            if sound < self.note_off_offset and sound >= self.note_on_offset:

                note = sound - self.note_on_offset
                note *= self.note_ratio
                msg = mido.Message('note_on', channel=0, note=note,
                                   velocity=self.vel_value, time=ticks)
                track.append(msg)

            # note_off
            elif sound < self.cc_offset:

                note = sound - self.note_off_offset
                note *= self.note_ratio
                msg = mido.Message('note_on', channel=0,
                                   note=note, velocity=0, time=ticks)
                track.append(msg)

            # control_change
            elif sound <= self.n_sounds:

                cc_idx = sound - self.cc_offset
                cc_control = self.cc_kept[cc_idx // 2]
                cc_value = self.cc_upper if cc_idx % 2 else self.cc_lower
                msg = mido.Message('control_change', channel=0,
                                   control=cc_control, value=cc_value, time=ticks)
                track.append(msg)

        end_of_track = mido.MetaMessage('end_of_track', time=ticks)
        track.append(end_of_track)

        midi = mido.MidiFile()
        midi.tracks.append(track)

        return midi

    def preprocess_dataset(self, src_filenames, dst_dir, batch_size, dst_filenames=None):

        assert len(src_filenames) >= batch_size
        if not dst_filenames is None:
            assert len(set(dst_filenames)) == len(src_filenames)
            assert re.findall('\/', ''.join(dst_filenames)) is None
            dst_filenames = [f if f.endswith(
                '.npz') else f + '.npz' for f in dst_filenames]
            dst_filenames = [os.path.join(dst_dir, f) for f in dst_filenames]
        else:
            dst_filenames = [os.path.join(dst_dir, str(
                f) + '.npz') for f in list(range(len(src_filenames)))]

        for idx in tqdm.tqdm(range(0, len(src_filenames), batch_size)):

            features_list = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self.midi_to_features)(f) for f in src_filenames[idx: idx + batch_size])

            for features, f in zip(features_list, dst_filenames[idx: idx + batch_size]):
                self.save_features(features, f)

    def get_tf_dataset(self, file_directory, batch_size, n_samples=None):

        filenames = sorted(glob.glob(os.path.join(file_directory, '*.npz')))
        assert len(filenames) > 0

        if n_samples:
            n_samples = min(n_samples, len(filenames))
            filenames = np.random.choice(
                filenames, n_samples, replace=False).tolist()

        buffer_size = len(filenames)

        #feature_list = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self.load_features)(file) for file in filenames)
        feature_list = [self.load_features(file) for file in filenames]
        sound_list = [x[0] for x in feature_list]
        delta_list = [x[1] for x in feature_list]
        sound_ragged = tf.ragged.constant(sound_list)
        delta_ragged = tf.ragged.constant(delta_list)

        dataset_sound = tf.data.Dataset.from_tensor_slices(sound_ragged)
        dataset_delta = tf.data.Dataset.from_tensor_slices(delta_ragged)

        tf_dataset = tf.data.Dataset.zip((dataset_sound, dataset_delta))
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (sound_ragged, delta_ragged))
        tf_dataset = tf_dataset.cache()
        tf_dataset = tf_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return tf_dataset

    def get_bigram_probs(self, npz_dir):
        '''
            Returns a matrix of conditional probablilties:
            P[D=d | S=s, S_prev=s_prev], where:
                D is the delta of the current timestamp
                S is the sound of the current timestamp
                S_prev is the sound of the previous timestamp

        '''

        npz_files = pathlib.Path(npz_dir).rglob('*.npz')
        npz_files = [str(f) for f in npz_files]
        assert len(npz_files) > 0

        freqs = np.zeros((self.n_sounds, self.n_sounds,
                          self.n_deltas), dtype=np.int32)

        sound_list, delta_list = zip(*list(map(self.load_features, npz_files)))

        for sounds, deltas in tqdm.tqdm(zip(sound_list, delta_list)):

            for idx, (sound, delta) in enumerate(zip(sounds[1:], deltas[1:])):

                freqs[sound, sounds[idx - 1], delta] += 1

        freqs_sum = np.sum(freqs, axis=-1)
        zero_mask = freqs_sum == 0
        freqs_sum[zero_mask] = 1
        freqs_sum = freqs_sum[:, :, np.newaxis]

        freqs_norm = freqs / freqs_sum

        return freqs_norm

    @staticmethod
    def build_from_config(config, idx_to_time):

        parser = MIDI_parser(tempo=config.tempo, ppq=config.ppq,
                             numerator=config.numerator, denominator=config.denominator,
                             clocks_per_click=config.clocks_per_click,
                             notated_32nd_notes_per_beat=config.notated_32nd_notes_per_beat,
                             cc_kept=config.cc_kept, cc_threshold=config.cc_threshold,
                             cc_lower=config.cc_lower, cc_upper=config.cc_upper,
                             n_notes=config.n_notes, n_deltas=config.n_deltas,
                             vel_value=config.vel_value, idx_to_time=idx_to_time,
                             n_jobs=config.n_jobs)

        return parser
