import pytest
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("..")) 
from dynamic_model.Debt import  EconomicNetwork, RiskFree, RiskDebt
from dynamic_model.Networks import BellmanNet_FOC



class DummyNet(EconomicNetwork):
    """A dummy network that provides deterministic outputs for testing."""
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Dense(4, activation=None)
        self.call_count = 0

    @tf.function
    def policy(self, K, B, Z):
        return 0.95 * K, 0.98 * B

    @tf.function
    def multiplier(self, K, B, Z):
        # 模拟一个稳定的 λ>0
        return 0.1 * tf.ones_like(K)

    @tf.function
    def value(self, K, B, Z):
        return 0.2 * K + 0.1 * B + 0.05 * Z

    def call(self, inputs):
        self.call_count += 1
        return self.layer(inputs)

@pytest.fixture
def model_free():
    return RiskFree(cost_type="Cost")

@pytest.fixture
def model_debt():
    dummy_prev = DummyNet()
    return RiskDebt(prev_net=dummy_prev, alpha=0.3)

@pytest.fixture
def dummy_net():
    return DummyNet()


def test_invalid_cost_type():
    """Invalid cost type"""
    with pytest.raises(ValueError):
        RiskFree(cost_type="ABC")

def test_profit_and_investment(model_free):
    K = tf.constant([[1.0]], dtype=tf.float32)
    Z = tf.constant([[2.0]], dtype=tf.float32)
    pi = model_free.profit(K, Z)
    assert np.isclose(pi.numpy(), 2 * 1.0 ** model_free.theta.numpy())

    Kp = tf.constant([[1.2]], dtype=tf.float32)
    inv = model_free.investment(Kp, K)
    expected = 1.2 - (1 - model_free.delta) * 1.0
    assert np.isclose(inv.numpy(), expected)

def test_investment_cost_switch(model_free):
    I = tf.constant([[0.1]], dtype=tf.float32)
    K = tf.constant([[1.0]], dtype=tf.float32)
    c1 = model_free.investment_cost(I, K)
    assert tf.reduce_all(c1 >= 0)

    m2 = RiskFree(cost_type="None")
    c2 = m2.investment_cost(I, K)
    assert np.allclose(c2.numpy(), 0.0)

def test_cashflow_and_reward(model_free):
    K = tf.constant([[1.0]])
    Kp = tf.constant([[1.1]])
    B = tf.constant([[0.2]])
    Bp = tf.constant([[0.25]])
    Z = tf.constant([[1.5]])

    e = model_free.cashflow(K, Kp, B, Bp, Z)
    eta = model_free.external_finance(e)
    r = model_free.Reward(None, K, Kp, B, Bp, Z)
    assert r.shape == (1, 1)
    assert tf.reduce_all(tf.math.is_finite(r))

def test_sample_state_and_lifetime(model_free, dummy_net):
    states = model_free.sample_state_train(5)
    for s in states:
        assert s.shape == (5, 1)

    rewards = model_free.lifetime_reward(dummy_net, states, T=3)
    assert rewards.shape == (5, 1)
    assert tf.reduce_all(tf.math.is_finite(rewards))


def test_bellman_residual_riskfree(model_free, dummy_net):
    states = model_free.sample_state_train(3)
    res = model_free.Bellman_residual(dummy_net, states)
    assert "loss_total" in res
    for k, v in res.items():
        assert v.dtype == tf.float32
        assert tf.rank(v) == 0  # scalar loss

def test_recovery_and_tilde(model_debt, dummy_net):
    K = tf.constant([[1.0]])
    Z = tf.constant([[1.0]])
    val = model_debt.recovery_value(K, Z)
    assert np.all(val.numpy() > 0)

    # Monte Carlo rate computation
    r_tilde = model_debt.get_r_tilde(dummy_net, K, tf.constant([[1.0]]), Z, N_mc=10)
    assert r_tilde.shape == (1, 1)
    assert tf.reduce_all(r_tilde >= -1.0)

def test_mask_and_tilde_test(model_debt, dummy_net):
    states = model_debt.sample_state_test(4)
    mask = model_debt.mask_default_set(dummy_net, *states)
    assert mask.shape == (4, 1)
    r_tilde = model_debt.get_r_tilde_test(dummy_net, states)
    assert tf.reduce_all(tf.math.is_finite(r_tilde))

def test_bellman_residual_riskdebt(model_debt, dummy_net):
    states = model_debt.sample_state_train(2)
    res = model_debt.Bellman_residual(dummy_net, states)
    assert isinstance(res, dict)
    assert "loss_total" in res
    assert tf.math.is_finite(res["loss_total"])