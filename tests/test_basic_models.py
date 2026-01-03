
import pytest
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("..")) 
from dynamic_model.Basic import  BasicModel
from dynamic_model.Networks import BellmanNet_FOC

tf.random.set_seed(1227)
np.random.seed(1227)

@pytest.fixture
def model():
    """Provide default initialized BasicModel."""
    return BasicModel()


@pytest.fixture
def dummy_states(model):
    """Provide deterministic state samples (K, Z) for small tests."""
    K = tf.constant([[float(model.K_steady)]], dtype=tf.float32)
    Z = tf.constant([[float(model.Z_min)]], dtype=tf.float32)
    return K, Z


class DummyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs):
        # return next K' and dummy V output shape consistent with expected
        out = self.dense(inputs)
        # duplicate so output shape (N, 2)
        return tf.concat([out, out], axis=1)

    @tf.function
    def policy(self, K, Z):
        return self(tf.concat([K, Z], 1))[:, :1]

    @tf.function
    def value(self, K, Z):
        return self(tf.concat([K, Z], 1))[:, 1:2]



def test_initialization_values(model):
    """Basic parameter setup sanity checks."""
    assert 0 < model.theta < 1
    assert 0 < model.delta < 1
    assert tf.reduce_all(model.K_min < model.K_max)
    assert np.isclose(float(model.beta), 1 / (1 + float(model.r)))


def test_invalid_cost_type():
    """Constructor should reject invalid cost_type names."""
    with pytest.raises(ValueError):
        BasicModel(cost_type="InvalidMode")


def test_profit_function(dummy_states, model):
    """profit(K, Z) == Z * K^theta"""
    K, Z = dummy_states
    numeric = model.profit(K, Z)
    expected = Z * tf.pow(K, model.theta)
    np.testing.assert_allclose(numeric.numpy(), expected.numpy(), rtol=1e-6)


def test_investment_function(dummy_states, model):
    """I = K' - (1-delta)*K"""
    Kp = tf.constant([[1.2], [2.0]], dtype=tf.float32)
    K, _ = dummy_states
    I = model.investment(Kp, K)
    expected = Kp - (1.0 - model.delta) * K
    np.testing.assert_allclose(I.numpy(), expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize("mode", ["None", "Cost"])
def test_investment_cost_shapes(model, mode):
    """investment_cost returns correct shapes and no NaNs."""
    m = BasicModel(cost_type=mode)
    I = tf.constant([[0.1], [0.5]], dtype=tf.float32)
    K = tf.constant([[2.0], [3.0]], dtype=tf.float32)
    cost = m.investment_cost(I, K)
    assert cost.shape == I.shape
    assert not tf.reduce_any(tf.math.is_nan(cost))


def test_cashflow_logic(model):
    """cashflow = profit - cost - investment"""
    K = tf.constant([[1.0]], dtype=tf.float32)
    Z = tf.constant([[1.0]], dtype=tf.float32)
    Kp = tf.constant([[1.1]], dtype=tf.float32)

    direct = model.cashflow(K, Z, Kp)
    manual = model.profit(K, Z) - model.investment(Kp, K) - model.investment_cost(model.investment(Kp, K), K)
    np.testing.assert_allclose(direct.numpy(), manual.numpy(), rtol=1e-7)


def test_state_sampling_ranges(model):
    """Sampled states fall within [K_min, K_max] and [Z_min, Z_max]."""
    K, Z = model.sample_state_train(1000)
    assert tf.reduce_all(K >= model.K_min)
    assert tf.reduce_all(K <= model.K_max)
    assert tf.reduce_all(Z >= model.Z_min)
    assert tf.reduce_all(Z <= model.Z_max)

    Kt, Zt = model.sample_state_test(100)
    assert tf.reduce_all(Kt >= model.K_min)
    assert tf.reduce_all(Kt <= model.K_max)
    assert tf.reduce_all(Zt >= model.Z_min)
    assert tf.reduce_all(Zt <= model.Z_max)


def test_lifetime_reward_runs_without_error(model):
    """Ensure lifetime_reward executes and outputs finite values."""
    net = DummyNet()
    Kt = tf.ones((8, 1))
    Zt = tf.ones((8, 1))
    total = model.lifetime_reward(net, (Kt, Zt), T=3)
    assert total.shape == (8, 1)
    assert tf.reduce_all(tf.math.is_finite(total))

def test_euler_residual_returns_scalar(model):
    """Euler_residual returns a scalar mean residual."""
    net = DummyNet()
    Kt = tf.ones((5, 1))
    Zt = tf.ones((5, 1))
    resid = model.Euler_residuals(net, (Kt, Zt))
    assert resid.shape == ()
    assert np.issubdtype(resid.numpy().dtype, np.floating)

def test_bellman_residual_output_dict(model):
    """Bellman_residual returns dictionary with correct keys and finite values."""
    net = DummyNet()
    K = tf.ones((6, 1))
    Z = tf.ones((6, 1))
    result = model.Bellman_residual(net, (K, Z))
    assert isinstance(result, dict)
    for key in ["loss_total", "loss_V", "loss_FOC"]:
        assert key in result
        assert tf.size(result[key]) == 1
        assert tf.reduce_all(tf.math.is_finite(result[key]))

def test_bellman_invalid_input_type_raises(model):
    """Bellman_residual should raise if cur_state is not tuple/list."""
    net = DummyNet()
    bad_input = tf.ones((4, 2))
    with pytest.raises(ValueError):
        model.Bellman_residual(net, bad_input)