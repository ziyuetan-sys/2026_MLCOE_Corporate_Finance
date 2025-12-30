
import pytest
import tensorflow as tf
from dynamic_model.networks import BellmanNet_FOC



@pytest.fixture
def net():
    model = "BasicModel" 
    return BellmanNet_FOC(model=model, hidden_dim=[8, 8], activation='relu')


def test_network_initialization(net):
   
    assert isinstance(net, tf.keras.Model)
    assert hasattr(net, "kp1") and hasattr(net, "v1")
    assert net.input_dim == 2
    assert net.output_dim == 2


def test_network_forward_output_shape(net):
    # 模拟输入 batch：5个样本, 每个2个特征 (K, Z)
    x = tf.random.normal((5, 2))
    y = net(x)

    assert isinstance(y, tf.Tensor)
    assert y.shape == (5, 2), "Output: (batch, 2)"
    # 分别为 K’ 和 V 的输出
    k_prime, v_val = tf.split(y, 2, axis=1)
    assert tf.reduce_all(k_prime >= 0), "K‘ > 0"


def test_initialize_weights(net):
    try:
        net.initialize_weights()
    except Exception as e:
        pytest.fail(f"initialize_weights() error: {e}")

    
    assert net.kp1.kernel is not None
    assert net.v2.bias is not None

def test_policy_method(net):
    K = tf.random.uniform((4, 1), minval=0.1, maxval=1.0)
    Z = tf.random.uniform((4, 1), minval=0.1, maxval=1.0)

    K_prime = net.policy(K, Z)

    assert isinstance(K_prime, tf.Tensor)
    assert K_prime.shape == (4, 1)
    assert tf.reduce_all(K_prime >= 0), "K' > 0"

def test_value_method(net):
    K = tf.random.uniform((4, 1))
    Z = tf.random.uniform((4, 1))

    V = net.value(K, Z)

    assert isinstance(V, tf.Tensor)
    assert V.shape == (4, 1)
    assert not tf.reduce_any(tf.math.is_nan(V)), "V is not NaN"


def test_graph_execution_consistency(net):
    K = tf.constant([[0.5], [1.0]], dtype=tf.float32)
    Z = tf.constant([[0.8], [1.2]], dtype=tf.float32)

    eager_out = net.policy(K, Z)

    @tf.function
    def graph_call():
        return net.policy(K, Z)

    graph_out = graph_call()
    diff = tf.reduce_mean(tf.abs(eager_out - graph_out)).numpy()

    assert diff < 1e-6, "The output of Eager and Graph mode should be same"