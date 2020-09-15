import joblib

tempo = 500000
ppq = 480
numerator = 4
denominator = 4
clocks_per_click = 24
notated_32nd_notes_per_beat = 8

cc_kept = [64, 67]
cc_threshold = 64
cc_lower = 0
cc_upper = 127

vel_value = 64

n_notes = 128

n_cc = 2 * len(cc_kept)
n_times = 122
n_classes = 2 * n_notes + n_cc + n_times + 1
pad_idx = 0
n_jobs = joblib.cpu_count()

seq_len = 512
mem_len = seq_len
d_model = 512
batch_size = 4
buffer_size = 420
n_heads = 4
dropout_rate = 0.3
n_layers = 10
n_epochs = 200
max_segs_per_batch = 20

lr = 0.00003

dataset_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip'
