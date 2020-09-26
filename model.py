import tensorflow as tf
import numpy as np
from utils import get_pos_encoding

__all__ = ('Music_transformer', 'Gated_Transformer_XL')


def gelu(x):

    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


tf.keras.utils.get_custom_objects().update(
    {'gelu': tf.keras.layers.Activation(gelu)})


class Relative_multi_head_attention(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, name='mha'):
        super(Relative_multi_head_attention, self).__init__(name=name)

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sqrt_dk = tf.math.sqrt(self.d_model * 1.0)

        self.w_q = tf.keras.layers.Dense(d_model, use_bias=False, name='w_q')
        self.w_k_e = tf.keras.layers.Dense(
            d_model, use_bias=False, name='w_k_e')
        self.w_k_r = tf.keras.layers.Dense(
            d_model, use_bias=False, name='w_k_r')
        self.w_v = tf.keras.layers.Dense(d_model, use_bias=False, name='w_v')

        self.final = tf.keras.layers.Dense(
            d_model, use_bias=False, activation='gelu', name='final')

        u_init = tf.random_normal_initializer()
        self.u_param = tf.Variable(
            initial_value=u_init(shape=(1, 1, self.n_heads, self.d_head), dtype="float32"), trainable=True, name='u_param')

        v_init = tf.random_normal_initializer()
        self.v_param = tf.Variable(
            initial_value=v_init(shape=(1, 1, self.n_heads, self.d_head), dtype="float32"), trainable=True, name='v_param'
        )

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

    def call(self, inputs, seq_len, mask, rel_enc):

        x_tilde = inputs

        # x_tilde -> (batch_size, full_len, d_model)
        # mem -> None or (batch_size, mem_len, d_model)
        # mask -> (1, 1, seq_len, mem_len + seq_len)
        # rel_enc -> (mem_len + seq_len, d_model)

        batch_size = x_tilde.shape[0]
        full_len = x_tilde.shape[1]

        x = x_tilde[:, -seq_len:, :]
        # x -> (batch_size, seq_len, d_model)

        full_len = x_tilde.shape[1]

        q = self.w_q(x)
        # q -> (batch_size, seq_len, d_model)
        k = self.w_k_e(x_tilde)
        # k -> (batch_size, full_len, d_model)
        v = self.w_v(x_tilde)
        # v -> (batch_size, full_len, d_model)

        q = tf.reshape(q, [batch_size, seq_len, self.n_heads, self.d_head])
        # q -> (batch_size, seq_len, n_heads, d_head)
        k = tf.reshape(k, [batch_size, full_len, self.n_heads, self.d_head])
        # k -> (batch_size, full_len, n_heads, d_head)
        v = tf.reshape(v, [batch_size, full_len, self.n_heads, self.d_head])
        # v -> (batch_size, full_len, n_heads, d_head)

        A_C = tf.einsum('bsnd,bfnd->bnsf', q + self.u_param, k)
        # A_C -> (batch_size, n_heads, seq_len, full_len)

        Q = self.w_k_r(rel_enc)
        # Q -> (full_len, d_model)
        Q = tf.reshape(Q, [full_len, self.n_heads, self.d_head])
        # Q -> (full_len, n_heads, d_head)

        B_D_hat = tf.einsum('bsnd, fnd->bnsf', q + self.v_param, Q)
        # B_D_hat -> (batch_size, n_heads, seq_len, full_len)

        B_D = self.rel_enc_shift(B_D_hat)
        # B_D -> (batch_size, n_heads, seq_len, full_len)

        attention_score = A_C + B_D
        attention_score = attention_score / self.sqrt_dk
        # attention_score -> (batch_size, n_heads, seq_len, full_len)

        attention_score += (mask * -1e10)

        attention_weights = tf.nn.softmax(attention_score, axis=-1)
        # attention_weights -> (batch_size, n_heads, seq_len, full_len)
        max_weights = tf.math.reduce_max(attention_weights, axis=-1)
        # max_weights -> (batch_size, n_heads, seq_len)
        max_weights = tf.math.reduce_max(max_weights, axis=-1)
        # max_weights -> (batch_size, n_heads)
        attention_loss = tf.math.reduce_mean(max_weights)

        attention_output = tf.einsum('bnsf,bfnd->bsnd', attention_weights, v)
        # attention_output -> (batch_size, seq_len, n_heads, d_head)
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model])
        # attention_output -> (batch_size, seq_len, d_model)

        output = self.final(attention_output)
        # output -> (batch_size, seq_len, d_model)

        return output, attention_weights, attention_loss


class Gating_layer_res(tf.keras.layers.Layer):

    def __init__(self):

        super(Gating_layer_res, self).__init__()

    def call(self, inputs):

        x, y = inputs

        return x + y


class Gating_layer_output(tf.keras.layers.Layer):

    def __init__(self, d_model):
        super(Gating_layer_output, self).__init__()

        self.d_model = d_model
        self.w_g = tf.keras.layers.Dense(
            self.d_model, activation='sigmoid', use_bias=True, name='w_g')

    def call(self, inputs):

        x, y = inputs

        # x -> (batch_size, seq_len, d_model)
        # y -> (batch_size, seq_len, d_model)

        g = x + self.w_g(x) * y

        return g


class Gating_layer_gru(tf.keras.layers.Layer):

    def __init__(self, d_model):
        super(Gating_layer_gru, self).__init__()

        self.d_model = d_model
        self.w_r = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='w_r')
        self.u_r = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='u_r')
        self.w_z = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='w_z')
        self.u_z = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='u_z')
        self.w_g = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='w_g')
        self.u_g = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='u_g')

        b_g_init = tf.zeros_initializer()
        self.b_g = tf.Variable(
            initial_value=b_g_init(shape=(d_model,), dtype="float32"), trainable=True, name='b_g')

    def call(self, inputs):

        x, y = inputs

        r = tf.keras.activations.sigmoid(self.w_r(y) + self.u_r(x))

        z = tf.keras.activations.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g)

        h_hat = tf.keras.activations.tanh(self.w_g(y) + self.u_g(r * x))

        g = (1 - z) * x + z * h_hat

        return g


class Transformer_block(tf.keras.layers.Layer):

    def __init__(self, d_model, n_heads, dropout_rate, gating_type=None):
        super(Transformer_block, self).__init__()

        assert 0.0 <= dropout_rate < 1

        self.d_model = d_model

        self.rmha = Relative_multi_head_attention(
            d_model=self.d_model, n_heads=n_heads)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.linear1 = tf.keras.layers.Dense(
            self.d_model, activation='gelu', name='linear1')
        self.linear2 = tf.keras.layers.Dense(
            self.d_model, activation='gelu', name='linear2')

        if gating_type == 'gru':

            self.gating_layer1 = Gating_layer_gru(self.d_model)
            self.gating_layer2 = Gating_layer_gru(self.d_model)

        elif gating_type == 'residual':

            self.gating_layer1 = Gating_layer_res()
            self.gating_layer2 = Gating_layer_res()

        else:

            self.gating_layer1 = Gating_layer_output(self.d_model)
            self.gating_layer2 = Gating_layer_output(self.d_model)

    def call(self, inputs, mem, mask, rel_enc, training):

        # inputs -> (batch_size, seq_len, d_model)
        # mem -> None or (batch_size, mem_len, d_model)
        # mask -> (1, 1, seq_len, mem_len + seq_len)
        # rel_enc -> (mem_len + seq_len, d_model)

        seq_len = inputs.shape[1]

        if mem is None:
            x_tilde = inputs
        else:
            x_tilde = tf.concat((tf.stop_gradient(mem), inputs), axis=1)
        # x_tilde -> (batch_size, full_len, d_model)

        x_tilde = self.layer_norm1(x_tilde)
        # attention_res -> (batch_size, seq_len, d_model)

        rmha_output, weight_list, attention_loss = self.rmha(
            x_tilde, seq_len, mask, rel_enc)
        rmha_output = self.dropout1(rmha_output, training=training)
        # rmha_output -> (batch_size, seq_len, d_model)

        rmha_output = self.gating_layer1((inputs, rmha_output))
        # rmha_output -> (batch_size, seq_len, d_model)

        output = self.layer_norm2(rmha_output)
        # output -> (batch_size, seq_len, d_model)
        output = self.linear1(output)
        # output -> (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        output = self.dropout2(output, training=training)
        # output -> (batch_size, seq_len, d_model)

        output = self.gating_layer2((rmha_output, output))

        return output, weight_list, attention_loss


class Music_transformer(tf.keras.Model):

    def __init__(self, d_sound, d_delta, n_heads_sound, n_heads_delta, n_heads_combined,
                 n_layers_sound, n_layers_delta, n_layers_combined,
                 n_sounds, n_deltas, dropout_rate, pad_idx,
                 weights_sound=None, weights_delta=None, max_seq_len=2048, gating_type=None):

        super(Music_transformer, self).__init__()

        assert d_sound % n_heads_sound == 0
        assert d_delta % n_heads_delta == 0
        assert (d_sound + d_delta) % n_heads_combined == 0
        assert 0.0 <= dropout_rate < 1.0

        self.d_sound = d_sound
        self.d_delta = d_delta
        self.d_combined = d_sound + d_delta
        self.n_heads_sound = n_heads_sound
        self.n_heads_delta = n_heads_delta
        self.n_heads_combined = n_heads_combined
        self.n_layers_sound = n_layers_sound
        self.n_layers_delta = n_layers_delta
        self.n_layers_combined = n_layers_combined
        self.n_layers_total = n_layers_sound + n_layers_delta + n_layers_combined
        self.n_sounds = n_sounds
        self.n_deltas = n_deltas
        self.dropout_rate = dropout_rate
        self.pad_idx = pad_idx

        if not weights_sound is None:
            weights_sound = tf.constant(weights_sound, dtype=tf.float32)
            assert weights_sound.shape == (self.n_sounds,)

        self.weights_sound = weights_sound

        if not weights_delta is None:
            weights_delta = tf.constant(weights_delta, dtype=tf.float32)
            assert weights_delta.shape == (self.n_deltas,)

        self.weights_delta = weights_delta

        self.emb_layer_sound = tf.keras.layers.Embedding(
            self.n_sounds, self.d_sound)
        self.emb_layer_delta = tf.keras.layers.Embedding(
            self.n_deltas, self.d_delta)
        self.pos_enc = get_pos_encoding(max_seq_len, self.d_combined)

        self.layer_list_sound = []
        for _ in range(self.n_layers_sound):

            layer = Transformer_block(
                self.d_sound, self.n_heads_sound, self.dropout_rate, gating_type)
            self.layer_list_sound.append(layer)

        self.layer_list_delta = []
        for _ in range(self.n_layers_delta):

            layer = Transformer_block(
                self.d_delta, self.n_heads_delta, self.dropout_rate, gating_type)
            self.layer_list_delta.append(layer)

        self.layer_list_combined = []
        for _ in range(self.n_layers_combined):

            layer = Transformer_block(
                self.d_combined, self.n_heads_combined, self.dropout_rate, gating_type)
            self.layer_list_combined.append(layer)

        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout1')
        self.hidden = tf.keras.layers.Dense(
            self.d_combined, activation='gelu', name='hidden')
        self.dropout2 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout2')
        self.final_sound = tf.keras.layers.Dense(
            self.n_sounds, name='final_sound')
        self.final_delta = tf.keras.layers.Dense(
            self.n_deltas, name='final_deltas')

    def get_look_ahead_mask(self, seq_len, mem_len):

        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # mask -> (seq_len, seq_len)
        if mem_len > 0:

            if mem_len < seq_len:
                mem_mask = 1 - tf.linalg.band_part(
                    tf.ones((seq_len, seq_len), dtype=mask.dtype), 0, -1)
                mem_mask = mem_mask[:, -mem_len:]
            else:
                mem_mask = 1 - tf.linalg.band_part(
                    tf.ones((seq_len, mem_len), dtype=mask.dtype), 0, -1)

            mask = tf.concat((mem_mask, mask), axis=-1)
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

        # sounds -> (batch_size, seq_len)
        # deltas -> (batch_size, seq_len)
        # mem_list : list of (batch_size, mem_len, d_model) or None
        # next_mem_len -> length of the next memory

        sounds, deltas = inputs

        batch_size = sounds.shape[0]
        seq_len = sounds.shape[1]

        if mem_list is None:
            mem_len = 0
            mem_list = [None] * self.n_layers_total
        else:
            mem_len = mem_list[0].shape[1]

        full_len = seq_len + mem_len

        mask = self.get_look_ahead_mask(seq_len, mem_len)
        # mask -> (1, 1, seq_len, full_len)

        rel_enc_sound = self.pos_enc[:full_len, :self.d_sound]
        rel_enc_sound = tf.reverse(rel_enc_sound, axis=[0])
        # rel_enc_sound -> (full_len, d_sound)

        rel_enc_delta = self.pos_enc[:full_len, :self.d_delta]
        rel_enc_delta = tf.reverse(rel_enc_delta, axis=[0])
        # rel_enc_delta -> (full_len, d_delta)

        rel_enc_combined = self.pos_enc[:full_len, :]
        rel_enc_combined = tf.reverse(rel_enc_combined, axis=[0])
        # rel_enc_combined -> (full_len, d_combined)

        next_mem_list = []
        attention_weight_list = []
        attention_loss_list = []

        sounds = self.emb_layer_sound(sounds)
        sounds = sounds * tf.math.sqrt(tf.cast(self.d_sound, tf.float32))
        # sounds -> (batch_size, seq_len, d_sound)

        for idx, layer in enumerate(self.layer_list_sound):

            next_mem = self.get_next_mem(mem_list[idx], sounds, next_mem_len)
            next_mem_list.append(next_mem)

            sounds, attention_weights, attention_loss = layer(
                sounds, mem_list[idx], mask, rel_enc_sound, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)
        # sounds -> (batch_size, seq_len, d_sound)

        deltas = self.emb_layer_delta(deltas)
        deltas = deltas * tf.math.sqrt(tf.cast(self.d_delta, tf.float32))
        # deltas -> (batch_size, seq_len, delta)

        for idx, layer in enumerate(self.layer_list_delta, self.n_layers_sound):

            next_mem = self.get_next_mem(mem_list[idx], deltas, next_mem_len)
            next_mem_list.append(next_mem)

            deltas, attention_weights, attention_loss = layer(
                deltas, mem_list[idx], mask, rel_enc_delta, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)
        # deltas -> (batch_size, seq_len, d_delta)

        x = tf.concat((sounds, deltas), axis=-1)

        for idx, layer in enumerate(self.layer_list_combined, self.n_layers_sound + self.n_layers_delta):

            next_mem = self.get_next_mem(mem_list[idx], x, next_mem_len)
            next_mem_list.append(next_mem)

            x, attention_weights, attention_loss = layer(
                x, mem_list[idx], mask, rel_enc_combined, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)
        # x -> (batch_size, seq_len, d_combined)

        x = self.dropout1(x, training=training)
        x = self.hidden(x)
        x = self.dropout2(x, training=training)
        # x -> (batch_size, seq_len, d_model)

        logits_sound = self.final_sound(x)
        # logits_sound -> (batch_size, seq_len, n_sounds)

        logits_delta = self.final_delta(x)
        # logits_sound -> (batch_size, seq_len, n_sounds)

        return logits_sound, logits_delta, next_mem_list, attention_weight_list, attention_loss_list

    def get_loss(self, logits_sound, logits_delta, labels_sound, labels_delta, attention_loss=None):

        # logits -> (batch_size, seq_len, n_classes)
        # labels -> (batch_size, seq_len)

        pad_mask_bool = tf.math.not_equal(labels_sound, self.pad_idx)
        pad_mask = tf.cast(pad_mask_bool, dtype=tf.float32)
        # pad_mask -> (batch_size, seq_len)

        num_not_padded = tf.math.reduce_sum(pad_mask)
        num_not_padded = tf.math.maximum(num_not_padded, 1.0)

        loss_sound = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_sound, logits=logits_sound)

        if not self.weights_sound is None:

            weights = tf.gather_nd(
                params=self.weights_sound, indices=labels_sound[..., tf.newaxis])
            loss_sound = loss_sound * weights

        loss_delta = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_delta, logits=logits_delta)

        if not self.weights_delta is None:

            weights = tf.gather_nd(
                params=self.weights_delta, indices=labels_delta[..., tf.newaxis])
            loss_delta = loss_delta * weights

        loss = loss_sound + loss_delta
        loss = loss * pad_mask
        # loss -> (batch_size, seq_len)

        loss = tf.math.reduce_sum(loss) / num_not_padded
        # loss -> ()

        if not attention_loss is None:

            loss += attention_loss

        return loss, pad_mask_bool

    @staticmethod
    def build_from_config(config, checkpoint_path=None, optimizer_path=None):

        model = Music_transformer(d_sound=config.d_sound, d_delta=config.d_delta,
                                  n_heads_sound=config.n_heads_sound, n_heads_delta=config.n_heads_delta,
                                  n_heads_combined=config.n_heads_combined, n_layers_sound=config.n_layers_sound,
                                  n_layers_delta=config.n_layers_delta, n_layers_combined=config.n_layers_combined,
                                  n_sounds=config.n_sounds, n_deltas=config.n_deltas,
                                  dropout_rate=config.dropout_rate, pad_idx=config.pad_idx)

        if not checkpoint_path is None:

            init_inputs = tf.zeros((4, 42), dtype=tf.int32)
            _ = model(inputs=(init_inputs, init_inputs), mem_list=None,
                      next_mem_len=None, training=False)

            model.load_weights(checkpoint_path)
            print(f'Loaded model weights from {checkpoint_path}')

        optimizer = tf.keras.optimizers.Adam(lr=config.lr)

        if not optimizer_path is None:

            optimizer_weights = np.load(optimizer_path, allow_pickle=True)
            grad_vars = model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))
            optimizer.set_weights(optimizer_weights)
            print(f'Loaded optimizer from {optimizer_path}')

        return model, optimizer


class Gated_Transformer_XL(tf.keras.Model):

    def __init__(self, d_model, n_heads, n_layers,
                 n_classes, dropout_rate, pad_idx,
                 class_weights=None, max_seq_len=2048, gating_type=None):

        super(Gated_Transformer_XL, self).__init__()

        assert d_model % n_heads == 0
        assert 0.0 <= dropout_rate < 1.0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.pad_idx = pad_idx

        if not class_weights is None:
            class_weights = tf.constant(class_weights, dtype=tf.float32)
            assert class_weights.shape == (self.n_classes,)

        self.class_weights = class_weights

        self.emb_layer = tf.keras.layers.Embedding(
            self.n_classes, self.d_model)
        self.pos_enc = get_pos_encoding(max_seq_len, self.d_model)

        self.layer_list = []
        for _ in range(self.n_layers):

            layer = Transformer_block(
                self.d_model, self.n_heads, self.dropout_rate, gating_type)
            self.layer_list.append(layer)

        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout1')
        self.hidden = tf.keras.layers.Dense(
            self.d_model, activation='gelu', name='hidden')
        self.dropout2 = tf.keras.layers.Dropout(
            self.dropout_rate, name='dropout2')
        self.final = tf.keras.layers.Dense(self.n_classes, name='final')

    def get_look_ahead_mask(self, seq_len, mem_len):

        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # mask -> (seq_len, seq_len)
        if mem_len > 0:

            if mem_len < seq_len:
                mem_mask = 1 - \
                    tf.linalg.band_part(
                        tf.ones((seq_len, seq_len), dtype=mask.dtype), 0, -1)
                mem_mask = mem_mask[:, -mem_len:]
            else:
                mem_mask = 1 - \
                    tf.linalg.band_part(
                        tf.ones((seq_len, mem_len), dtype=mask.dtype), 0, -1)

            mask = tf.concat((mem_mask, mask), axis=-1)
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
            mem_list = [None] * self.n_layers
        else:
            mem_len = mem_list[0].shape[1]

        full_len = seq_len + mem_len

        mask = self.get_look_ahead_mask(seq_len, mem_len)
        # mask -> (1, 1, seq_len, full_len)

        rel_enc = self.pos_enc[:full_len, :]
        rel_enc = tf.reverse(rel_enc, axis=[0])
        # rel_enc -> (full_len, d_model)

        next_mem_list = []
        attention_weight_list = []
        attention_loss_list = []

        x = self.emb_layer(inputs)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x -> (batch_size, seq_len, d_model)

        for idx, layer in enumerate(self.layer_list):

            next_mem = self.get_next_mem(mem_list[idx], x, next_mem_len)
            next_mem_list.append(next_mem)

            x, attention_weights, attention_loss = layer(
                x, mem_list[idx], mask, rel_enc, training)
            attention_weight_list.append(attention_weights)
            attention_loss_list.append(attention_loss)
        # x -> (batch_size, seq_len, d_model)

        x = self.dropout1(x, training=training)
        x = self.hidden(x)
        x = self.dropout2(x, training=training)
        # x -> (batch_size, seq_len, d_model)

        logits = self.final(x)
        # logits -> (batch_size, seq_len, n_classes)

        return logits, next_mem_list, attention_weight_list, attention_loss_list

    def get_loss(self, logits, labels, attention_loss=None):

        # logits -> (batch_size, seq_len, n_classes)
        # labels -> (batch_size, seq_len)

        pad_mask_bool = tf.math.not_equal(labels, self.pad_idx)
        pad_mask = tf.cast(pad_mask_bool, dtype=tf.float32)
        # pad_mask -> (batch_size, seq_len)

        num_not_padded = tf.math.reduce_sum(pad_mask)
        num_not_padded = tf.math.maximum(num_not_padded, 1.0)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        if not self.class_weights is None:

            class_weights = tf.gather_nd(
                params=self.class_weights, indices=labels[..., tf.newaxis])
            loss = loss * class_weights

        loss = loss * pad_mask
        # loss -> (batch_size, seq_len)

        loss = tf.math.reduce_sum(loss) / num_not_padded
        # loss -> ()

        if not attention_loss is None:

            loss += attention_loss

        return loss, pad_mask_bool

    @staticmethod
    def build_from_config(config, checkpoint_path=None, optimizer_path=None):

        model = Gated_Transformer_XL(d_model=config.d_model, n_heads=config.n_heads,
                                     n_layers=config.n_layers, n_classes=config.n_classes,
                                     dropout_rate=config.dropout_rate, pad_idx=config.pad_idx)

        if not checkpoint_path is None:

            init_inputs = tf.zeros((4, 42), dtype=tf.int32)
            _ = model(inputs=init_inputs, mem_list=None,
                      next_mem_len=None, training=False)

            model.load_weights(checkpoint_path)
            print(f'Loaded model weights from {checkpoint_path}')

        optimizer = tf.keras.optimizers.Adam(lr=config.lr)

        if not optimizer_path is None:

            optimizer_weights = np.load(optimizer_path, allow_pickle=True)
            grad_vars = model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))
            optimizer.set_weights(optimizer_weights)
            print(f'Loaded optimizer from {optimizer_path}')

        return model, optimizer
