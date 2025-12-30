
import tensorflow as tf
from tensorflow.keras import layers, Model

class BellmanNet_FOC(Model):
    """
    Bellman AiO Network: can share or separate subnets for K' and V outputs.

    """
    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):
        super().__init__()
        self.model = model
        self.activation = activation
        self.hidden_dim = hidden_dim
       
        self.input_dim = 2     # [K, Z]
        self.output_dim = 2    # [K_prime, V]
       
        self.kp1 = layers.Dense(hidden_dim[0], activation=activation)
        self.kp2 = layers.Dense(hidden_dim[1], activation=activation)
        self.kp_out = layers.Dense(1, activation=None)
      
        self.v1 = layers.Dense(hidden_dim[0], activation=activation)
        self.v2 = layers.Dense(hidden_dim[1], activation=activation)
        self.v_out = layers.Dense(1, activation=None)
 

    def initialize_weights(self):
        he_init = tf.keras.initializers.HeNormal(seed = 1227)
        small_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed = 1227)
        zero_init = tf.keras.initializers.Zeros()

        def init_layer(layer, out_layer=False):
            if out_layer:
                layer.kernel.assign(small_init(shape=layer.kernel.shape))
            else:
                layer.kernel.assign(he_init(shape=layer.kernel.shape))
            layer.bias.assign(zero_init(shape=layer.bias.shape))


        _ = self(tf.zeros((1, self.input_dim))) 
        # K subnetwork
        init_layer(self.kp1)
        init_layer(self.kp2)
        init_layer(self.kp_out, out_layer=True)
        # V subnetwork
        init_layer(self.v1)
        init_layer(self.v2)
        init_layer(self.v_out, out_layer=True)


    def call(self, inputs):
        k = self.kp1(inputs)
        k = self.kp2(k)
        k_out = self.kp_out(k)
        k_out = tf.nn.softplus(k_out)

        v = self.v1(inputs)
        v = self.v2(v)
        v_out = self.v_out(v)

        out = tf.concat([k_out, v_out], axis=1)
        return out
    
    @tf.function
    def policy(self, K, Z):
        """Compute next-period capital K_prime given states (K, Z)"""
        inputs = tf.concat([K, Z], axis=1)
        out = self(inputs)
        K_prime =out[:, :1]  # positive output
        return K_prime

    @tf.function
    def value(self, K, Z):
        """Compute current value function V(K,Z)"""
        inputs = tf.concat([K, Z], axis=1)
        out = self(inputs)
        V_now = out[:, 1:2]
        return V_now


class BellmanNet_RiskyFree(Model):
    """
    Bellman  Network: can share or separate subnets for K' and V outputs.
    """
    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):#, mode='split'):
        super().__init__()
        self.model = model
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.input_dim = 3     # [K, B, Z]
        self.output_dim = 4    # [K_prime, B_prime, lamda,  V]

        self.input_norm = layers.LayerNormalization(axis=-1)
        self.kp1 = layers.Dense(hidden_dim[0], activation=activation)
        self.kp2 = layers.Dense(hidden_dim[1], activation=activation)
        self.kp_out = layers.Dense(1, activation=None)

        self.l1 = layers.Dense(hidden_dim[0], activation=activation)
        self.l2 = layers.Dense(hidden_dim[1], activation=activation)
        self.l_out = layers.Dense(1, activation=None)   

        self.bp1 = layers.Dense(hidden_dim[0], activation=activation)
        self.bp2 = layers.Dense(hidden_dim[1], activation=activation)
        self.bp_out = layers.Dense(1, activation=None)
     
        self.v1 = layers.Dense(hidden_dim[0], activation=activation)
        self.v2 = layers.Dense(hidden_dim[1], activation=activation)
        self.v_out = layers.Dense(1, activation=None)
    
    def initialize_weights(self):
        he_init = tf.keras.initializers.HeNormal(seed = 1227)
        small_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed = 1227)
        zero_init = tf.keras.initializers.Zeros()

        _ = self(tf.ones((1, self.input_dim), dtype=tf.float32))

        def init_layer(layer, out_layer=False):
            if out_layer:
                layer.kernel.assign(small_init(shape=layer.kernel.shape))
            else:
                layer.kernel.assign(he_init(shape=layer.kernel.shape))
            layer.bias.assign(zero_init(shape=layer.bias.shape))

        for ly in [
            self.kp1, self.kp2,
            self.bp1, self.bp2,
            self.l1, self.l2,
            self.v1, self.v2]:
            init_layer(ly)

        for ly in [self.kp_out, self.bp_out, self.l_out, self.v_out]:
            init_layer(ly, out_layer=True)

    def call(self, inputs):
    
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != 3:
                raise ValueError(f"Expected (K, B, Z), got {len(inputs)} elements.")
            K, B, Z = inputs
            inputs = tf.concat([K, B, Z], axis=1)
        elif isinstance(inputs, tf.Tensor):
            if inputs.shape[-1] != 3:
                raise ValueError(f"Expected inputs.shape[-1]=3, got {inputs.shape[-1]}")
            K, B, Z = tf.split(inputs, num_or_size_splits=3, axis=1)
        else:
            raise TypeError(f"inputs must be tuple/list of tensors or a single tf.Tensor")
        
        k = self.input_norm(K)
        k = self.kp1(k)
        k = self.kp2(k)
        k_out = self.kp_out(k)
        K_prime = tf.nn.softplus(k_out) # Ensure output K_prime > 0
      

        b = self.bp1(inputs)
        b = self.bp2(inputs)
        b_out = self.bp_out(b)
        B_min  = self.model.B_min
        B_max = self.model.collateral_constraint(K_prime)
        B_prime = B_min + tf.nn.sigmoid(b_out) * (B_max - B_min)    

        lam = self.l1(inputs)
        lam = self.l2(lam)
        lam_out = self.l_out(lam)
        Lam = tf.nn.softplus(lam_out)  #ebsure lambda > 0
       
        v = self.v1(inputs)
        v = self.v2(v)
        V_now = self.v_out(v)

        out = tf.concat([K_prime, B_prime, Lam, V_now], axis=1)  # [batch,4]
        return out
    
    def policy(self, K, B, Z):
        """
        Compute next-period (K', B') given current (K, B, Z)
        """
        inputs = tf.concat([K, B, Z], axis=1)
        out = self(inputs)
        K_prime = out[:, :1]
        B_prime = out[:, 1:2]
        return K_prime, B_prime

    def multiplier(self, K, B, Z):
        """Compute current period multiplier Lam(K,B,Z)"""
        inputs = tf.concat([K, B, Z], axis=1)
        out = self(inputs)
        Lam = out[:, 2:3]
        return Lam

   
    def value(self, K, B, Z):
        """Compute current value function V(K,B,Z)"""
        inputs = tf.concat([K, B, Z], axis=1)
        out = self(inputs)
        V_now = out[:, 2:3]
        return V_now
    
class BellmanNet_RiskDebt(Model):
    def __init__(self, model, hidden_dim=[64, 64], activation='relu'):
        super().__init__()
        self.model = model
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.input_dim = 3     # [K, B, Z]
        self.output_dim = 3    # [K_prime, B_prime, V]

        self.input_norm = layers.LayerNormalization(axis=-1)
        self.kp1 = layers.Dense(hidden_dim[0], activation=activation)
        self.kp2 = layers.Dense(hidden_dim[1], activation=activation)
        self.kp_out = layers.Dense(1, activation=None)

        self.bp1 = layers.Dense(hidden_dim[0], activation=activation)
        self.bp2 = layers.Dense(hidden_dim[1], activation=activation)
        self.bp_out = layers.Dense(1, activation=None)
     
        self.v1 = layers.Dense(hidden_dim[0], activation=activation)
        self.v2 = layers.Dense(hidden_dim[1], activation=activation)

        self.v_out = layers.Dense(1, activation=None)

    def initialize_weights(self):
        he_init = tf.keras.initializers.HeNormal(seed = 1227)
        small_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed = 1227)
        zero_init = tf.keras.initializers.Zeros()

        def init_layer(layer, out_layer=False):
            if out_layer:
                layer.kernel.assign(small_init(shape=layer.kernel.shape))
            else:
                layer.kernel.assign(he_init(shape=layer.kernel.shape))
            layer.bias.assign(zero_init(shape=layer.bias.shape))

            _ = self(tf.zeros((1, self.input_dim))) 
            for ly in [self.kp1, self.kp2, self.bp1, self.bp2, self.v1, self.v2]:
                init_layer(ly)
            for ly in [self.kp_out, self.bp_out, self.v_out]:
                init_layer(ly, out_layer=True)



    def call(self, inputs):
        if isinstance(inputs, (tuple, list)):
            K, B, Z = inputs
            inputs = tf.concat([K, B, Z], axis=1)
        K, B, Z = tf.split(inputs, 3, axis=1)

        # Capital update branch
        k = self.kp1(K)
        k = self.kp2(k)
        K_prime = tf.nn.softplus(self.kp_out(k))

        # Debt update branch (无 B_max 限制)
        b_in = self.bp1(inputs)
        b_in = self.bp2(b_in)
        lam = tf.nn.sigmoid(self.bp_out(b_in))   # ∈ (0,1)
        B_min = self.model.B_min
        B_max = self.model.B_max # No collateral constraint
        B_prime = B_min + lam * (B_max - B_min)

        # Value branch
        v = self.v1(inputs)
        v = self.v2(v)
        V_now = self.v_out(v)

        return tf.concat([K_prime, B_prime, V_now], axis=1)

    def policy(self, K, B, Z):
        out = self.call((K, B, Z))
        return out[:, :1], out[:, 1:2]

    def value(self, K, B, Z):
        out = self.call((K, B, Z))
        return out[:, 2:3]
