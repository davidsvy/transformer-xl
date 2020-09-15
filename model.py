import tensorflow as tf
import numpy as np
from utils import get_pos_encoding


def gelu(x):

    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


tf.keras.utils.get_custom_objects().update(
    {'gelu': tf.keras.layers.Activation(gelu)})


class Multi_head_attention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name='mha'):
        super(Multi_head_attention, self).__init__(name=name)

        assert isinstance(d_model, int)
        assert isinstance(num_heads, int)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.sqrt_dk = tf.constant(
            tf.cast(tf.math.sqrt(self.d_head * 1.0), dtype=tf.float32))

        self.w_q = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), name='w_q')
        self.w_k_e = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), name='w_k_e')
        self.w_k_r = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), name='w_k_r')
        self.w_v = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), name='w_v')

        self.linear = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), name='linear')
        '''
        u_init = tf.random_normal_initializer()
        self.u_param = tf.Variable(
            initial_value=u_init(shape=(self.d_head, 1), dtype="float32"),
            trainable=True,
        )
        '''
        '''
        v_init = tf.random_normal_initializer()
        self.v_param = tf.Variable(
            initial_value=v_init(shape=(self.d_head, 1), dtype="float32"),
            trainable=True, name='v_param'
        )
        '''

    def split_heads(self, x, batch_size):
        """
        input dims -> (batch_size, seq_len, d_model)
        output dims -> (batch_size, num_heads, seq_len, d_head)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        # x -> (batch_size, seq_len, num_heads, d_head)
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # x -> (batch_size, num_heads, seq_len, d_head)

    def split_heads_rel(self, x):
        """
        input dims -> (seq_len, d_model)
        output dims -> (num_heads, seq_len, d_head)
        """
        x = tf.reshape(x, (x.shape[0], self.num_heads, self.d_head))
        # x -> (seq_len, num_heads, d_head)
        x = tf.transpose(x, perm=[1, 0, 2])
        # x -> (num_heads, seq_len, d_head)
        return x

    def rel_enc_shift(self, arr):
        """
        input dims -> (seq_len, num_heads, l, m)
        output dims -> (seq_len, num_heads, l, m)
        """

        batch_size, num_heads, l, m = arr.shape
        zeros = tf.zeros((batch_size, num_heads, l, 1), dtype=arr.dtype)
        arr = tf.concat((arr, zeros), axis=-1)
        arr = tf.reshape(arr, [batch_size, num_heads, -1])
        arr = tf.reshape(arr[:, :, l-1: -1], [batch_size, num_heads, l, m])
        return arr

    def call(self, x, mem, mask, rel_enc):

        # x -> (batch_size, seq_len, d_model)
        # mem -> None or (batch_size, mem_len, d_model)
        # mask -> (1, 1, seq_len, mem_len + seq_len)
        # rel_enc -> (mem_len + seq_len, d_model)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not mem is None:
            x_tilde = tf.concat((mem, x), axis=1)
        else:
            x_tilde = x
        # x_tilde -> (batch_size, cur_mem_len + seq_len, d_model)

        q = self.w_q(x)
        # q -> (batch_size, seq_len, d_model)
        k = self.w_k_e(x_tilde)
        # k -> (batch_size, cur_mem_len + seq_len, d_model)
        v = self.w_v(x_tilde)
        # v -> (batch_size, cur_mem_len + seq_len, d_model)

        q = self.split_heads(q, batch_size)
        # q -> (batch_size, num_heads, seq_len, d_head)
        k = self.split_heads(k, batch_size)
        # k -> (batch_size, num_heads, cur_mem_len + seq_len, d_head)
        v = self.split_heads(v, batch_size)
        # v -> (batch_size, num_heads, cur_mem_len + seq_len, d_head)

        A = tf.matmul(q, k, transpose_b=True)
        # A -> (batch_size, num_heads, seq_len, cur_mem_len + seq_len)

        Q = self.w_k_r(rel_enc)
        # Q -> (mem_len + seq_len, d_model)
        Q = self.split_heads_rel(Q)
        # Q -> (num_heads, mem_len + seq_len, d_head)

        B_hat = tf.matmul(q, Q, transpose_b=True)
        # B_hat -> (batch_size, num_heads, seq_len, mem_len + seq_len)

        B = self.rel_enc_shift(B_hat)
        # B -> (batch_size, num_heads, seq_len, mem_len + seq_len)

        #C = tf.matmul(k, self.u_param)
        # C -> (batch_size, num_heads, mem_len + seq_len, 1)

        '''
        D_hat = tf.matmul(Q, self.v_param)
        # D_hat -> (num_heads, mem_len + seq_len, 1)
        D_hat = tf.transpose(D_hat, perm=[0, 2, 1])
        # D_hat -> (num_heads, 1, mem_len + seq_len)
        D_hat = tf.tile(D_hat, [1, seq_len, 1])
        # D_hat -> (num_heads, seq_len, mem_len + seq_len)
        D_hat = D_hat[tf.newaxis, ...]
        # D_hat -> (1, num_heads, seq_len, mem_len + seq_len)

        D = self.rel_enc_shift(D_hat)
        # D -> (1, num_heads, seq_len, mem_len + seq_len)
        '''

        attention_score = A + B
        #attention_score += D
        # attention_score -> (batch_size, num_heads, seq_len, mem_len + seq_len)

        attention_score_scaled = attention_score / self.sqrt_dk
        # attention_score_scaled -> (batch_size, num_heads, seq_len, mem_len + seq_len)

        if not mask is None:
            attention_score_scaled += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_score_scaled, axis=-1)
        # attention_weights -> (batch_size, num_heads, seq_len, mem_len + seq_len)

        attention_output = tf.matmul(attention_weights, v)
        # attention_output -> (batch_size, num_heads, seq_len, d_head)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # attention_output -> (batch_size, seq_len, num_heads, d_head)
        attention_output = tf.reshape(
            attention_output, (batch_size, seq_len, self.d_model))
        # attention_output -> (batch_size, seq_len, d_model)

        output = self.linear(attention_output)
        # output -> (batch_size, seq_len, d_model)

        return output


class Transformer_block(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dropout_rate):
        super(Transformer_block, self).__init__()
        assert isinstance(d_model, int)
        assert isinstance(num_heads, int)
        assert 0.0 <= dropout_rate < 1

        self.multi_head_attention = Multi_head_attention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = tf.keras.layers.Dense(
            d_model, input_shape=(d_model,), activation='gelu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.linear2 = tf.keras.layers.Dense(d_model, input_shape=(d_model,))
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, mem, mask, rel_enc, training):

        # inputs -> (batch_size, seq_len, d_model)
        # mem -> None or (batch_size, mem_len, d_model)
        # mask -> (1, 1, seq_len, mem_len + seq_len)
        # rel_enc -> (mem_len + seq_len, d_model)

        attention = self.multi_head_attention(inputs, mem, mask, rel_enc)
        # attention -> (batch_size, seq_len, d_model)
        attention = self.dropout1(attention, training=training)
        # attention -> (batch_size, seq_len, d_model)
        attention_res = self.layer_norm1(inputs + attention)
        # attention_res -> (batch_size, seq_len, d_model)

        linear_out = self.linear1(attention_res)
        # linear_out -> (batch_size, seq_len, d_model)
        linear_out = self.linear2(linear_out)
        # linear_out -> (batch_size, seq_len, d_model)
        linear_out = self.dropout2(linear_out, training=training)
        # linear_out -> (batch_size, seq_len, d_model)
        outputs = self.layer_norm2(attention_res + linear_out)
        # outputs -> (batch_size, seq_len, d_model)

        return outputs


class Transformer_XL(tf.keras.Model):

    def __init__(self, d_model, num_heads, dropout_rate, num_layers, num_classes, pad_idx,
                 class_weights=None, max_seq_len=2048):

        super(Transformer_XL, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pad_idx = pad_idx

        if not class_weights is None:
            class_weights = np.array(class_weights)
            assert class_weights.shape == (self.num_classes,)
            class_weights = tf.constant(class_weights, dtype=tf.float32)

        self.class_weights = class_weights

        self.emb_layer = tf.keras.layers.Embedding(
            self.num_classes, self.d_model)
        self.pos_enc = get_pos_encoding(max_seq_len, self.d_model)

        self.block_list = [Transformer_block(
            self.d_model, self.num_heads, self.dropout_rate) for _ in range(self.num_layers)]

        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout1')
        self.hidden = tf.keras.layers.Dense(self.d_model, input_shape=(
            self.d_model,), activation='gelu', name='hidden')
        self.dropout2 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout2')
        self.final_dense = tf.keras.layers.Dense(
            self.num_classes, input_shape=(self.d_model,), name='final_dense')

    def get_look_ahead_mask(self, seq_len, mem_len):

        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # mask -> (seq_len, seq_len)
        if mem_len > 0:
            paddings = [[0, 0], [mem_len, 0]]
            mask = tf.pad(mask, paddings=paddings)
            # mask -> (seq_len, mem_len + seq_len)

        mask = mask[tf.newaxis, tf.newaxis, :, :]
        # mask -> (1, 1, seq_len, mem_len + seq_len)

        return mask

    def get_next_mem(self, prev_mem, next_mem, next_mem_len):

        # prev_mem -> None or (batch_size, mem_len, d_model)
        # next_mem -> (batch_size, seq_len, d_model)

        if prev_mem is None or next_mem_len is None or next_mem_len == 0:
            res = next_mem
            # res -> (batch_size, seq_len, d_model)
        else:
            res = tf.concat((prev_mem, next_mem), axis=1)[
                :, -(next_mem_len):, :]
            # res -> (batch_size, next_mem_len, d_model)

        res = tf.stop_gradient(res)

        return res

    def call(self, inputs, mem_list, next_mem_len, training):

        # inputs -> (batch_size, seq_len)
        # mem_list : list of (batch_size, mem_len, d_model) or None
        # next_mem_len -> length of the next memory

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        if mem_list is None:
            mem_len = 0
            mem_list = [None] * self.num_layers
        else:
            mem_len = mem_list[0].shape[1]

        mask = self.get_look_ahead_mask(seq_len, mem_len)
        # mask -> (1, 1, seq_len, mem_len + seq_len)

        rel_enc = self.pos_enc[:(mem_len + seq_len), :]
        rel_enc = tf.reverse(rel_enc, axis=[0])
        # rel_enc -> (mem_len + seq_len, d_model)

        x = self.emb_layer(inputs)
        # x -> (batch_size, seq_len, d_model)

        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x -> (batch_size, seq_len, d_model)

        # This will contain the input of each layer in encoder_list
        next_mem_list = []

        for idx, layer in enumerate(self.block_list):

            next_mem = self.get_next_mem(mem_list[idx], x, next_mem_len)
            next_mem_list.append(next_mem)

            x = layer(x, mem_list[idx], mask, rel_enc, training)
        # x -> (batch_size, seq_len, d_model)

        x = self.dropout1(x, training=training)
        x = self.hidden(x)
        x = self.dropout2(x, training=training)
        # x -> (batch_size, seq_len, d_model)

        logits = self.final_dense(x)
        # pitch_logits -> (batch_size, seq_len, num_classes)

        return logits, next_mem_list

    def get_loss(self, logits, labels):

        # logits -> (batch_size, seq_len, num_classes)
        # labels -> (batch_size, seq_len)

        pad_mask_bool = tf.math.not_equal(labels, self.pad_idx)
        pad_mask = tf.cast(pad_mask_bool, dtype=tf.float32)
        # pad_mask -> (batch_size, seq_len)

        num_not_padded = tf.math.reduce_sum(pad_mask)
        num_not_padded = tf.math.maximum(num_not_padded, 1.0)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        if not self.class_weights is None:

            weights = tf.gather_nd(
                params=self.class_weights, indices=labels[..., tf.newaxis])
            loss = loss * weights

        #loss = tf.where(pad_mask, loss, 0)
        loss = loss * pad_mask
        # loss -> (batch_size, seq_len)

        loss = tf.math.reduce_sum(loss) / num_not_padded
        # loss -> ()

        return loss, pad_mask_bool

    def get_output(self, inputs, mem_len, training, mask):

        # inputs -> (batch_size, seq_len)
        # mask -> (batch_size, 1, seq_len, mem_len + seq_len)

        logits, next_mem_list = self.call(inputs, mem_len, training, mask)
        # logits -> (batch_size, seq_len, num_classes)

        outputs = tf.nn.softmax(logits, axis=-1)
        # outputs -> (batch_size, seq_len, num_classes)

        return outputs, next_mem_list

    @staticmethod
    def build_from_config(config, checkpoint_path=None):

        if hasattr(config, 'n_classes'):
            n_classes = config.n_classes
        elif hasattr(config, 'n_words'):
            n_classes = config.n_words
        else:
            raise Exception('Config does not contain the number of classes')

        model = Transformer_XL(d_model=config.d_model,
                               num_heads=config.n_heads,
                               dropout_rate=config.dropout_rate,
                               num_layers=config.n_layers,
                               num_classes=n_classes,
                               pad_idx=config.pad_idx)

        if not checkpoint_path is None:

            init_inputs = tf.zeros((4, 42), dtype=tf.int32)
            _ = model(inputs=init_inputs, mem_list=None,
                      next_mem_len=None, training=False)

            model.load_weights(checkpoint_path)
            print(f'Loaded weights from {checkpoint_path}')

        return model
