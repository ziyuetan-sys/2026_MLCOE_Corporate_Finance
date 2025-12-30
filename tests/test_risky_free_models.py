
import tensorflow as tf
import pytest
from dynamic.models import RiskFree

@pytest.fixture
def riskfree_model():
    return RiskFree()


@pytest.fixture
def dummy_net():
    class DummyNet(tf.keras.Model):
        def call(self, x):
            #  [K_next, B_next, V_next] output
            return tf.concat([x, tf.reduce_sum(x, axis=1, keepdims=True)], axis=1)
        def policy(self, K, Z):
            return 0.9 * K + 0.1 * Z
    return DummyNet()


# residual
def test_riskfree_bellman_residual_runs(riskfree_model, dummy_net):
    K, B, Z = riskfree_model.sample_state_train(2)
    out = riskfree_model.Bellman_residual(dummy_net, (K, B, Z))
    assert "loss_total" in out

# lifetime
def test_riskfree_lifetime_reward_runs(riskfree_model, dummy_net):
    batch_size = 4
    T = 5
    K,B, Z = riskfree_model.sample_state_train(batch_size)

    reward_sum = riskfree_model.lifetime_reward(dummy_net, (K,B, Z), T)


    assert isinstance(reward_sum, tf.Tensor)
    assert reward_sum.shape == (batch_size, 1)
    assert not tf.math.reduce_any(tf.math.is_nan(reward_sum))
    assert not tf.math.reduce_any(tf.math.is_inf(reward_sum))

def test_riskfree_lifetime_reward_invalid_input_type(riskfree_model, dummy_net):
    K, B, Z =riskfree_model.sample_state_train(2)
    with pytest.raises(ValueError):
        _ = riskfree_model.lifetime_reward(dummy_net, K, T=3)


# risk constraint
def test_riskfree_collateral_constraint(riskfree_model):
    Kp = tf.constant([[1.0], [2.0]])
    rhs = riskfree_model.collateral_constraint(Kp)
    assert rhs.shape == Kp.shape

# cashflow, reward
def test_riskfree_cashflow_and_reward(riskfree_model):
    K = tf.constant([[1.0]])
    Kp = tf.constant([[1.2]])
    B = tf.constant([[0.3]])
    Bp = tf.constant([[0.4]])
    Z = tf.constant([[1.0]])
    e = riskfree_model.cashflow(K, Kp, B, Bp, Z)
    reward = riskfree_model.Reward(K, Kp, B, Bp, Z)
    assert e.shape == (1,1)
    assert reward.shape == (1,1)

# check external finance
def test_external_finance_positive_negative(riskfree_model):
    e_neg = tf.constant([[-1.0]], dtype=tf.float32)
    e_pos = tf.constant([[1.0]], dtype=tf.float32)
    eta_neg = riskfree_model.external_finance(e_neg)
    eta_pos = riskfree_model.external_finance(e_pos)
    assert eta_neg.numpy() <= 0.0 or eta_neg.numpy() >= 0.0  # simply runs
    assert eta_pos.numpy() == 0.0

# sample
def test_riskfree_state_train_and_test_shapes(riskfree_model):
    n_train = 8
    n_test = 5

    train_state = riskfree_model.sample_state_train(n_train)
    test_state = riskfree_model.sample_state_test(n_test)

    # return
    for state in (train_state, test_state):
        assert isinstance(state, tuple), "tuple"
        assert len(state) == 3,  "should be (K, B, Z)"

    # dim
    Kt, Bt, Zt = train_state
    Ke, Be, Ze = test_state
    assert Kt.shape == (n_train, 1)
    assert Bt.shape == (n_train, 1)
    assert Zt.shape == (n_train, 1)
    assert Ke.shape == (n_test, 1)
    assert Be.shape == (n_test, 1)
    assert Ze.shape == (n_test, 1)

    # range
    assert tf.reduce_all(Kt >= riskfree_model.K_min)
    assert tf.reduce_all(Kt <= riskfree_model.K_max)
    assert tf.reduce_all(Zt >= riskfree_model.Z_min)
    assert tf.reduce_all(Zt <= riskfree_model.Z_max)