
import tensorflow as tf
from tensorflow.keras import layers, Model


def dense_layer (hidden_dim, activation):
    """a two-layer MLP block and output layer."""
    return (
        layers.Dense(hidden_dim[0], activation=activation),
        layers.Dense(hidden_dim[1], activation=activation),
        layers.Dense(1, activation=None),
    )


def init_all_layers(model, hidden_layers, output_layers):
    """Initialize weights of layers using HeNormal and TruncatedNormal."""
    he_init = tf.keras.initializers.HeNormal(seed=1227)
    small_init = tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1227)
    zero_init = tf.keras.initializers.Zeros()

    def _init(layer, out=False):
        layer.kernel.assign((small_init if out else he_init)(layer.kernel.shape))
        layer.bias.assign(zero_init(layer.bias.shape))

    _ = model(tf.zeros((1, model.input_dim)))  # Build layers
    for ly in hidden_layers:
        _init(ly)
    for ly in output_layers:
        _init(ly, out=True)


def _concat_inputs(inputs, expected_dim):
    """Concatenate tuple/list inputs or validate tensor."""
    if isinstance(inputs, (tuple, list)):
        inputs = tf.concat(inputs, axis=1)
    elif isinstance(inputs, tf.Tensor) and inputs.shape[-1] != expected_dim:
        raise ValueError(f"Expected tensor with last dim={expected_dim}")
    return inputs


class BellmanNet_FOC(Model):
    """Bellman AiO Network: jointly approximates policy (K') and value (V)."""

    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):
        super().__init__()
        self.model, self.hidden_dim, self.activation = model, hidden_dim, activation
        self.input_dim, self.output_dim = 2, 2
        self.kp1, self.kp2, self.kp_out = dense_layer(hidden_dim, activation)
        self.v1, self.v2, self.v_out = dense_layer(hidden_dim, activation)

    def initialize_weights(self):
        init_all_layers(self, [self.kp1, self.kp2, self.v1, self.v2], [self.kp_out, self.v_out])

    def call(self, inputs):
        k = tf.nn.softplus(self.kp_out(self.kp2(self.kp1(inputs))))
        v = self.v_out(self.v2(self.v1(inputs)))
        return tf.concat([k, v], axis=1)

    @tf.function
    def policy(self, K, Z): return self(tf.concat([K, Z], 1))[:, :1]

    @tf.function
    def value(self, K, Z): return self(tf.concat([K, Z], 1))[:, 1:2]


class BellmanNet_RiskyFree(Model):
    """Risk-free model predicting [K', B', λ, V]."""

    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):
        super().__init__()
        self.model, self.hidden_dim, self.activation = model, hidden_dim, activation
        self.input_dim, self.output_dim = 3, 4
        self.input_norm = layers.LayerNormalization(axis=-1)
        self.kp1, self.kp2, self.kp_out = dense_layer(hidden_dim, activation)
        self.bp1, self.bp2, self.bp_out = dense_layer(hidden_dim, activation)
        self.l1, self.l2, self.l_out = dense_layer(hidden_dim, activation)
        self.v1, self.v2, self.v_out = dense_layer(hidden_dim, activation)

    def initialize_weights(self):
        hidden_layers = [self.kp1, self.kp2, self.bp1, self.bp2, self.l1, self.l2, self.v1, self.v2]
        outputs = [self.kp_out, self.bp_out, self.l_out, self.v_out]
        init_all_layers(self, hidden_layers, outputs)

    def call(self, inputs):
        inputs = _concat_inputs(inputs, 3)
        K, B, Z = tf.split(inputs, 3, axis=1)

        Kp = tf.nn.softplus(self.kp_out(self.kp2(self.kp1(self.input_norm(K)))))
        B_min, B_max = self.model.B_min, self.model.collateral_constraint(Kp)
        Bp = B_min + tf.nn.sigmoid(self.bp_out(self.bp2(self.bp1(inputs)))) * (B_max - B_min)
        Lam = tf.nn.softplus(self.l_out(self.l2(self.l1(inputs))))
        V = self.v_out(self.v2(self.v1(inputs)))

        return tf.concat([Kp, Bp, Lam, V], axis=1)

    def policy(self, K, B, Z): out = self(tf.concat([K, B, Z], 1)); return out[:, :1], out[:, 1:2]
    def multiplier(self, K, B, Z): return self(tf.concat([K, B, Z], 1))[:, 2:3]
    def value(self, K, B, Z): return self(tf.concat([K, B, Z], 1))[:, 3:4]


class BellmanNet_RiskDebt(Model):
    """Risky debt model predicting [K', B', V] (no collateral constraint)."""

    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):
        super().__init__()
        self.model, self.hidden_dim, self.activation = model, hidden_dim, activation
        self.input_dim, self.output_dim = 3, 3
        self.input_norm = layers.LayerNormalization(axis=-1)
        self.kp1, self.kp2, self.kp_out = dense_layer(hidden_dim, activation)
        self.bp1, self.bp2, self.bp_out = dense_layer(hidden_dim, activation)
        self.v1, self.v2, self.v_out = dense_layer(hidden_dim, activation)

    def initialize_weights(self):
        hidden_layers = [self.kp1, self.kp2, self.bp1, self.bp2, self.v1, self.v2]
        outputs = [self.kp_out, self.bp_out, self.v_out]
        init_all_layers(self, hidden_layers, outputs)

    def call(self, inputs):
        inputs = _concat_inputs(inputs, 3)
        K, _, _ = tf.split(inputs, 3, axis=1)

        Kp = tf.nn.softplus(self.kp_out(self.kp2(self.kp1(K))))
        u = tf.nn.sigmoid(self.bp_out(self.bp2(self.bp1(inputs))))
        B_min, B_max = self.model.B_min, self.model.B_max
        Bp = B_min + u * (B_max - B_min)
        V = self.v_out(self.v2(self.v1(inputs)))

        return tf.concat([Kp, Bp, V], axis=1)

    def policy(self, K, B, Z): out = self((K, B, Z)); return out[:, :1], out[:, 1:2]
    def value(self, K, B, Z): return self((K, B, Z))[:, 2:3]