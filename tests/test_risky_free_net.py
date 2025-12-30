import pytest
import tensorflow as tf
from dynamic_model.networks import BellmanNet_RiskyFree


@pytest.fixture
def dummy_model():
    class DummyModel:
        def __init__(self):
            self.K_min = 0.1
            self.K_max = 10.0
            self.K_steady = 1.0

        def collateral_constraint(self, K_prime):
            return (0.8 * K_prime)
    return DummyModel()


@pytest.fixture
def net(dummy_model):
    tf.random.set_seed(42)
    net = BellmanNet_RiskyFree(model=dummy_model, hidden_dim=[8, 8])
    return net

def test_riskyfree_net_initialization(net):
    assert isinstance(net, tf.keras.Model)
    assert net.input_dim == 3
    assert net.output_dim == 3
    assert hasattr(net, "kp1") and hasattr(net, "bp1") and hasattr(net, "v1")
    assert callable(net.call)


def test_forward_output_shape(net, dummy_model):
    x = tf.random.normal((4, 3))  # [K, B, Z]
    y = net(x)

    assert isinstance(y, tf.Tensor)
    assert y.shape == (4, 3), "output (batch, 3)"
    Kp, Bp, V = tf.split(y, 3, axis=1)
    Bmax = dummy_model.collateral_constraint(Kp)
    assert tf.reduce_all(Kp >= 0), "K' > 0"
    assert tf.reduce_all(Bp <= Bmax + 1e-6), "B' satisfies collateral constraint"
    assert not tf.reduce_any(tf.math.is_nan(V)), "V is not NaN"


def test_initialize_weights(net):
   
    try:
        net.initialize_weights()
    except Exception as e:
        pytest.fail(f"initialize_weights initialze error: {e}")

    assert net.kp1.kernel is not None
    assert net.v2.bias is not None


def test_policy_and_value(net, dummy_model):
    K = tf.random.uniform((3, 1), 0.1, 2.0)
    B = tf.random.uniform((3, 1), -0.5, 1.5)
    Z = tf.random.uniform((3, 1), 0.8, 1.2)

    Kp, Bp = net.policy(K, B, Z)
    V = net.value(K, B, Z)
    Bmax = dummy_model.collateral_constraint(Kp)
    # 类型 & 形状检查
    assert isinstance(Kp, tf.Tensor) and Kp.shape == (3, 1)
    assert isinstance(Bp, tf.Tensor) and Bp.shape == (3, 1)
    assert isinstance(V, tf.Tensor) and V.shape == (3, 1)
    # 数值合理性
    assert tf.reduce_all(Kp >= 0), "K' > 0"
    assert tf.reduce_all(Bp <= Bmax + 1e-6), "B' satisfies collateral constraint"


def test_graph_execution_consistency(net):
    K = tf.constant([[1.0], [0.5]])
    B = tf.constant([[0.2], [0.8]])
    Z = tf.constant([[1.1], [0.9]])

    eager_out = net.policy(K, B, Z)

    @tf.function
    def graph_call():
        return net.policy(K, B, Z)

    graph_out = graph_call()
    diff = tf.reduce_mean(tf.abs(tf.concat(eager_out, axis=1) - tf.concat(graph_out, axis=1)))

    assert diff < 1e-6,  "The output of Eager and Graph mode should be same"


def test_batch_consistency(net):
    for batch_size in [1, 8, 16]:
        K = tf.random.uniform((batch_size, 1))
        B = tf.random.uniform((batch_size, 1))
        Z = tf.random.uniform((batch_size, 1))
        Kp, Bp = net.policy(K, B, Z)
        V = net.value(K, B, Z)
        assert Kp.shape == (batch_size, 1)
        assert Bp.shape == (batch_size, 1)
        assert V.shape == (batch_size, 1)