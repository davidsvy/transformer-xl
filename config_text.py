n_classes = 6000  # n_words
d_model = 384
seq_len = 128
mem_len = 2 * seq_len
batch_size = 4
n_heads = 6
dropout_rate = 0.3
n_layers = 8
n_epochs = 200

lr = 0.00002
max_segs_per_batch = 8

pad_idx = 0

email_token = '00email00'
url_token = '00url00'
money_token = '00money00'
tel_token = '00tel00'
name_token = '00name00'
relative_token = '00relative00'
start_token = '00start00'
end_token = '00end00'
unknown_token = '00unknown00'
