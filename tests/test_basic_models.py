# test_basic_model.py
import pytest
import tensorflow as tf
from model import BasicModel  # your model file name

@pytest.fixture
def model():
    """Fixture to initialize a BasicModel instance for tests."""
    return BasicModel()

@pytest.fixture
def sample_states(model):
    """Fixture to generate sample states."""
    return model.sample_state_train(5)

def test_initialization(model):
    """Check steady-state initialization and basic parameter bounds."""
    assert 0 < model.theta < 1
    assert 0 < model.delta < 1
    assert isinstance(model.K_steady, tf.Tensor)
    assert model.K_steady.numpy() > 0

def test_profit(model):
    """Test the profit function for monotonicity in K."""
    K = tf.constant([[1.0], [2.0], [4.0]], dtype=tf.float32)
    Z = tf.constant([[1.0], [1.0], [1.0]], dtype=tf.float32)
    profit_vals = model.profit(K, Z).numpy()
    assert profit_vals[0] < profit_vals[-1]

def test_investment_and_cost(model):
    """Ensure investment and cost computations behave as expected."""
    K = tf.constant([[10.0]], dtype=tf.float32)
    K_prime = tf.constant([[11.0]], dtype=tf.float32)
    I = model.investment(K_prime, K)
    assert I.numpy().item() == pytest.approx(10.0 * model.delta.numpy() + 1.0, rel=0.1)

    psi_none = model.investment_cost(I, K)
    assert psi_none.numpy().shape == I.numpy().shape

def test_cashflow(model):
    """Test that cashflow decreases with higher investment costs."""
    K = tf.constant([[10.0]], dtype=tf.float32)
    Z = tf.constant([[1.0]], dtype=tf.float32)
    K_prime_low = tf.constant([[10.2]], dtype=tf.float32)
    K_prime_high = tf.constant([[12.0]], dtype=tf.float32)
    cf_low = model.cashflow(K, Z, K_prime_low)
    cf_high = model.cashflow(K, Z, K_prime_high)
    assert cf_high.numpy() < cf_low.numpy()

def test_sample_state_train(model):
    """Ensure training samples fall within specified bounds."""
    K, Z = model.sample_state_train(100)
    assert tf.reduce_all((K >= model.K_min) & (K <= model.K_max))
    assert tf.reduce_all((Z >= model.Z_min) & (Z <= model.Z_max))

def test_value_residual_computation(model, sample_states):
    """Check that Bellman residual returns expected dictionary keys."""
    class DummyNet:
        def policy(self, K, Z): return K * 0.95
        def value(self, K, Z): return K * 0.1
        def __call__(self, x): return tf.concat([x[:, :1] * 0.1], axis=1)

    net = DummyNet()
    residuals = model.Bellman_residual(net, sample_states)
    assert "loss_total" in residuals
    assert "loss_V" in residuals
    assert "loss_FOC" in residuals