
import tensorflow as tf
import pytest
from dynamic.models import BasicModel


@pytest.fixture
def basic_model():
    return BasicModel()


@pytest.fixture
def dummy_net():
    class DummyNet(tf.keras.Model):
        def call(self, x):
            #  [K_next, B_next, V_next] output
            return tf.concat([x, tf.reduce_sum(x, axis=1, keepdims=True)], axis=1)
        def policy(self, K, Z):
            return 0.9 * K + 0.1 * Z
    return DummyNet()



# profit, investment, output
def test_basicmodel_profit_and_investment(basic_model):
    K = tf.constant([[1.0], [2.0]], dtype=tf.float32)
    Z = tf.constant([[1.0], [0.5]], dtype=tf.float32)
    profit = basic_model.profit(K, Z)
    invest = basic_model.investment(K_prime=1.5 * K, K=K)
    assert profit.shape == K.shape
    assert invest.shape == K.shape

# cashflow
def test_basicmodel_cashflow(basic_model):
    K = tf.constant([[1.0]])
    Z = tf.constant([[1.0]])
    K_prime = tf.constant([[1.2]])
    cashflow = basic_model.cashflow(K, Z, K_prime)
    assert cashflow.shape == (1, 1)

# lifetime
def test_basicmodel_lifetime_reward_runs(basic_model, dummy_net):
    batch_size = 4
    T = 5
    K, Z = basic_model.sample_state_train(batch_size)

    reward_sum = basic_model.lifetime_reward(dummy_net, (K, Z), T)


    assert isinstance(reward_sum, tf.Tensor)
    assert reward_sum.shape == (batch_size, 1)
    assert not tf.math.reduce_any(tf.math.is_nan(reward_sum))
    assert not tf.math.reduce_any(tf.math.is_inf(reward_sum))

def test_basicmodel_lifetime_reward_invalid_input_type(basic_model, dummy_net):
    K, Z = basic_model.sample_state_train(2)
    with pytest.raises(ValueError):
        _ = basic_model.lifetime_reward(dummy_net, K, T=3)

      
# bellman residual
def test_basicmodel_bellman_residual_runs(basic_model, dummy_net):
    K, Z = basic_model.sample_state_train(4)
    out = basic_model.Bellman_residual(dummy_net, (K, Z))
    assert "loss_total" in out and isinstance(out["loss_total"], tf.Tensor)

# sample data
def test_basicmodel_sample_state_train_and_test_shapes(basic_model):
    n_train = 8
    n_test = 5

    train_state = basic_model.sample_state_train(n_train)
    test_state = basic_model.sample_state_test(n_test)

    # return
    for state in (train_state, test_state):
        assert isinstance(state, tuple), "return must bbe tuple"
        assert len(state) == 2, "should be (K, Z)"
    # dim
    Kt, Zt = train_state
    Ke, Ze = test_state
    assert Kt.shape == (n_train, 1)
    assert Zt.shape == (n_train, 1)
    assert Ke.shape == (n_test, 1)
    assert Ze.shape == (n_test, 1)

    # range
    assert tf.reduce_all(Kt >= basic_model.K_min)
    assert tf.reduce_all(Kt <= basic_model.K_max)
    assert tf.reduce_all(Zt >= basic_model.Z_min)
    assert tf.reduce_all(Zt <= basic_model.Z_max)
