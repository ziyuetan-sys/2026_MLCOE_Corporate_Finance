
import pytest
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import os
import sys
sys.path.append(os.path.abspath("..")) 
from dynamic_model.Debt import RiskFree, RiskDebt
from dynamic_model.Networks import BellmanNet_RiskyFree, BellmanNet_RiskDebt, BellmanNet_FOC
from dynamic_model.Networks import dense_layer, _concat_inputs, init_all_layers



class MockModel:
    """Mock model with properties required for networks."""
    def __init__(self):
        self.B_min = tf.constant(-1.0)
        self.B_max = tf.constant(2.0)

    def collateral_constraint(self, Kp):
        # Simple linear constraint
        return tf.clip_by_value(1.5 * Kp, -2.0, 3.0)

@pytest.fixture
def dummy_inputs():
    """Provide simple test inputs."""
    return tf.random.normal((8, 3))


def test_dense_layer_output_shapes():
    """dense_layer should return tuple of 3 Dense layers."""
    hidden = [32, 16]
    activation = "relu"
    layers_tuple = dense_layer(hidden, activation)

    assert len(layers_tuple) == 3
    assert all(isinstance(l, layers.Dense) for l in layers_tuple)
    assert layers_tuple[0].units == hidden[0]
    assert layers_tuple[-1].units == 1


def test_concat_inputs_valid_tuple(dummy_inputs):
    """_concat_inputs should concat tensor tuple correctly along axis=1."""
    split_inputs = tf.split(dummy_inputs, 3, axis=1)
    out = _concat_inputs(split_inputs, expected_dim=3)
    assert isinstance(out, tf.Tensor)
    assert out.shape[1] == 3


def test_concat_inputs_invalid_dim_raises(dummy_inputs):
    """Should raise ValueError when tensor has unexpected last dimension."""
    wrong_input = tf.random.normal((10, 5))
    with pytest.raises(ValueError):
        _concat_inputs(wrong_input, expected_dim=3)

def test_init_all_layers_initializes_weights():
    """init_all_layers should assign new kernel/bias values."""
    layers_tuple = dense_layer([8, 4], "relu")
    model = tf.keras.Sequential(layers_tuple)
    model.input_dim = 3


    model(tf.zeros((1, model.input_dim)))

    before = [ly.get_weights() for ly in model.layers]
    init_all_layers(model, model.layers[:-1], [model.layers[-1]])
    after = [ly.get_weights() for ly in model.layers]

    for (bk, bb), (ak, ab) in zip(before, after):
        assert len(bk) == ak.shape[0] or bk.shape == ak.shape  
        assert not np.array_equal(bk, ak)

def test_bellmannet_foc_forward_output_shape(dummy_inputs):
    """BellmanNet_FOC forward pass should return (N,2)."""
    net = BellmanNet_FOC(model=None)
    out = net(dummy_inputs[:, :2])  # input_dim=2
    assert out.shape == (8, 2)


def test_bellmannet_foc_policy_value(dummy_inputs):
    """Verify FOC.policy and FOC.value outputs correct shapes."""
    net = BellmanNet_FOC(model=None)
    K = dummy_inputs[:, :1]
    Z = dummy_inputs[:, 1:2]
    kpolicy = net.policy(K, Z)
    vvalue = net.value(K, Z)
    assert kpolicy.shape == (8, 1)
    assert vvalue.shape == (8, 1)


def test_bellmannet_riskyfree_forward_output_shape(dummy_inputs):
    """BellmanNet_RiskyFree forward pass outputs (N,4)."""
    mock_model = MockModel()
    net = BellmanNet_RiskyFree(mock_model)
    out = net(dummy_inputs)
    assert out.shape == (8, 4)


def test_bellmannet_riskyfree_policy_multiplier_value(dummy_inputs):
    """RiskyFree network: check forward outputs are numeric."""
    mock_model = MockModel()
    net = BellmanNet_RiskyFree(mock_model)
    K, B, Z = tf.split(dummy_inputs, 3, axis=1)
    policy_out = net.policy(K, B, Z)
    lam = net.multiplier(K, B, Z)
    val = net.value(K, B, Z)
    assert isinstance(policy_out[0], tf.Tensor)
    assert policy_out[0].shape[1] == 1
    assert lam.shape == (8, 1)
    assert val.shape == (8, 1)

def test_riskyfree_output_b_within_constraints():
    """Ensure RiskyFree network B' output satisfies model constraints."""
    mock_model = MockModel()
    net = BellmanNet_RiskyFree(mock_model)
    batch_size = 16

    K = tf.random.uniform((batch_size, 1), minval=0.0, maxval=2.0)
    B = tf.random.uniform((batch_size, 1), minval=-1.0, maxval=1.0)
    Z = tf.random.uniform((batch_size, 1), minval=-0.5, maxval=0.5)
    inputs = tf.concat([K, B, Z], axis=1)

    out = net(inputs)
    assert out.shape == (batch_size, 4)

    Kp, Bp = out[:, 0:1], out[:, 1:2]

    B_max = mock_model.collateral_constraint(Kp)

    Bp_val, Bmax_val = Bp.numpy(), B_max.numpy()

    # Check B' <= B_max
    assert np.all(Bp_val <= Bmax_val + 1e-6), f"B' above upper bound! max diff={(Bp_val - Bmax_val).max()}"


def test_bellmannet_riskdebt_forward_output_shape(dummy_inputs):
    """BellmanNet_RiskDebt forward should output (N,3)."""
    mock_model = MockModel()
    net = BellmanNet_RiskDebt(mock_model)
    out = net(dummy_inputs)
    assert out.shape == (8, 3)


def test_bellmannet_riskdebt_policy_value(dummy_inputs):
    """Check separate interfaces for policy and value outputs."""
    mock_model = MockModel()
    net = BellmanNet_RiskDebt(mock_model)
    K, B, Z = tf.split(dummy_inputs, 3, axis=1)
    kp, bp = net.policy(K, B, Z)
    val = net.value(K, B, Z)
    assert kp.shape == (8, 1)
    assert bp.shape == (8, 1)
    assert val.shape == (8, 1)


def test_initialize_weights_executes_without_error():
    """initialize_weights should successfully execute all initializers."""
    mock_model = MockModel()
    net = BellmanNet_RiskyFree(mock_model)
    net.initialize_weights()  # Should not raise
    # Check all layer weights exist and have proper shapes
    for ly in net.layers:
        if hasattr(ly, "kernel"):
            assert isinstance(ly.kernel, (tf.Variable))
        if hasattr(ly, "bias"):
            assert isinstance(ly.bias, (tf.Variable))


