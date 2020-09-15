import tensorflow as tf
import numpy as np
import mido
import re
import os
import joblib
import glob
import tqdm
from collections import Counter


class MIDI_parser():

    def __init__(self, tempo, ppq, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat,
                 cc_kept, cc_threshold, cc_lower, cc_upper, n_notes, n_times, vel_value, idx_to_time, n_jobs):

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

        self.n_notes = n_notes
        self.n_cc = 2 * len(self.cc_kept)
        self.n_times = n_times
        self.n_classes = 2 * self.n_notes + self.n_cc + self.n_times + 1

        self.pad_idx = 0
        self.n_jobs = n_jobs

        assert n_times == len(idx_to_time)
        assert idx_to_time[0] == 0
        assert np.sum(idx_to_time == 0) == 1

        self.idx_to_time = idx_to_time

        self.note_on_offset = 1
        self.note_off_offset = self.note_on_offset + self.n_notes
        self.cc_offset = self.note_off_offset + self.n_notes
        self.time_offset = self.cc_offset + self.n_cc

        self.time_range = (self.time_offset, self.n_classes)
        self.sound_range = (self.note_on_offset, self.time_offset)

    def secs_to_ticks(self, secs):

        return int(round(1e6 * self.ppq / self.tempo * secs))

    def save_features(self, features, filename):

        np.save(filename, features)

    def load_features(self, filename):

        return np.load(filename)

    def midi_to_features(self, src_file):

        midi = mido.MidiFile(src_file)
        encoded = []

        for msg in midi:

            if msg.time == 0:
                time = 0
            else:
                time = 1 + np.clip(np.digitize(msg.time,
                                               self.idx_to_time[1:]), 0, self.n_times - 2)
            time += self.time_offset

            # note on
            if msg.type == 'note_on' and msg.velocity > 0:

                note_on = msg.note
                note_on += self.note_on_offset

                encoded.append(time)
                encoded.append(note_on)

            # note_off
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):

                note_off = msg.note
                note_off += self.note_off_offset

                encoded.append(time)
                encoded.append(note_off)

            # control_change
            elif msg.type == 'control_change' and msg.control in self.cc_kept:

                control_idx = self.cc_kept.index(msg.control)
                value = msg.value >= self.cc_threshold
                cc = control_idx * 2 + value
                cc += self.cc_offset

                encoded.append(time)
                encoded.append(cc)

        return np.array(encoded).astype(np.uint16)

    def features_to_midi(self, features):

        track = mido.MidiTrack()

        tempo = mido.MetaMessage('set_tempo', tempo=self.tempo, time=0)
        time_signature = mido.MetaMessage('time_signature', numerator=self.numerator, denominator=self.denominator,
                                          clocks_per_click=self.clocks_per_click,
                                          notated_32nd_notes_per_beat=self.notated_32nd_notes_per_beat, time=0)

        track.append(tempo)
        track.append(time_signature)

        mask = features != self.pad_idx
        features = features[mask]

        prev_time = 0

        for feature in features:

            # note_on
            if feature < self.note_off_offset and feature >= self.note_on_offset:

                note = feature - self.note_on_offset
                msg = mido.Message('note_on', channel=0, note=note,
                                   velocity=self.vel_value, time=prev_time)
                track.append(msg)

            # note_off
            elif feature < self.cc_offset:

                note = feature - self.note_off_offset
                msg = mido.Message('note_on', channel=0,
                                   note=note, velocity=0, time=prev_time)
                track.append(msg)

            # control_change
            elif feature < self.time_offset:

                cc_idx = feature - self.cc_offset
                cc_control = self.cc_kept[cc_idx // 2]
                cc_value = self.cc_upper if cc_idx % 2 else self.cc_lower
                msg = mido.Message(
                    'control_change', channel=0, control=cc_control, value=cc_value, time=prev_time)
                track.append(msg)

            # time_shift
            elif feature < self.n_classes:

                time_idx = feature - self.time_offset
                secs = self.idx_to_time[time_idx]
                ticks = self.secs_to_ticks(secs)
                prev_time = ticks

        end_of_track = mido.MetaMessage('end_of_track', time=prev_time)
        track.append(end_of_track)

        midi = mido.MidiFile()
        midi.tracks.append(track)

        return midi

    def preprocess_dataset(self, src_filenames, dst_dir, batch_size, dst_filenames=None):

        assert len(src_filenames) >= batch_size
        if dst_filenames:
            assert len(set(dst_filenames)) == len(src_filenames)
            assert not re.findall(r'\/', ''.join(dst_filenames))
            dst_filenames = [f if f.endswith(
                '.npy') else f + '.npy' for f in dst_filenames]
            dst_filenames = [os.path.join(dst_dir, f) for f in dst_filenames]
        else:
            dst_filenames = [os.path.join(dst_dir, str(
                f) + '.npy') for f in list(range(len(src_filenames)))]

        for idx in tqdm.tqdm(range(0, len(src_filenames), batch_size)):

            features_list = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self.midi_to_features)(f) for f in src_filenames[idx: idx + batch_size])

            for features, f in zip(features_list, dst_filenames[idx: idx + batch_size]):
                self.save_features(features, f)

    def get_tf_dataset(self, file_directory, batch_size, buffer_size, n_samples=None):

        filenames = sorted(glob.glob(os.path.join(file_directory, '*.npy')))
        assert len(filenames) > 0

        if n_samples:
            assert isinstance(n_samples, int)
            n_samples = min(n_samples, len(filenames))
            filenames = np.random.choice(
                filenames, n_samples, replace=False).tolist()

        feature_list = [self.load_features(file) for file in filenames]
        features_ragged = tf.ragged.constant(feature_list)

        tf_dataset = tf.data.Dataset.from_tensor_slices((features_ragged))
        tf_dataset = tf_dataset.cache()
        tf_dataset = tf_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return tf_dataset

    def get_class_weights(self, file_directory, T=1.4):

        class_counter = Counter()
        npy_filenames = sorted(
            glob.glob(os.path.join(file_directory, '*.npy')))
        assert len(npy_filenames) > 0

        for file in tqdm.tqdm_notebook(npy_filenames):

            arr = np.load(file)
            class_counter.update(arr)

        class_freqs = np.zeros((self.n_classes), dtype=np.int32)
        for class_, freq in class_counter.items():
            class_freqs[class_] = freq

        eps = 1e-5

        zero_mask = class_freqs == 0

        # smoothe
        smoothed_freqs = np.exp(np.log(1 + class_freqs) / T)
        # normalize
        smoothed_probs = smoothed_freqs / np.sum(smoothed_freqs)

        smoothed_weights = 1 / (eps + smoothed_probs)
        smoothed_weights[zero_mask] = 0
        norm_coef = np.sum(smoothed_weights * smoothed_probs)
        final_weights = smoothed_weights / norm_coef

        return final_weights
