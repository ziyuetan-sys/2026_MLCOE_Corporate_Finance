# test_bellman_trainer.py
import pytest
import tensorflow as tf
import numpy as np
from types import SimpleNamespace
import os
import sys
sys.path.append(os.path.abspath("..")) 
from dynamic_model.Basic import BasicModel
from dynamic_model.Debt import RiskFree, RiskDebt
from dynamic_model.Networks import BellmanNet_RiskyFree, BellmanNet_RiskDebt, BellmanNet_FOC
from dynamic_model.Trainer import BellmanTrainer

# Mock net
class MockNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs[0])

    def initialize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(tf.random.normal(layer.kernel.shape))

class MockModel:
    def Bellman_residual(self, net, state_batch, nu):
        x = state_batch[0]
        loss_value = tf.reduce_mean(x ** 2) + tf.constant(nu, dtype=tf.float32) * 1e-4
        return {"loss_total": loss_value, "loss_aux": loss_value * 
        0.5}

    def lifetime_reward(self, net, test_data, T=200):
        x = test_data[0]
        return tf.reduce_mean(x) + tf.constant(1.0, dtype=tf.float32)

    def sample_state_test(self, test_size):
        return (tf.random.normal((test_size, 3)), tf.random.normal((test_size, 3)))

    def sample_state_train(self, batch_size):
        return (tf.random.normal((batch_size, 3)), tf.random.normal((batch_size, 3)))


@pytest.fixture
def mock_trainer():
    """Fixture to create a BellmanTrainer instance with mock dependencies."""
    mock_model = MockModel()
    mock_net = MockNet()

    train_data = (
        tf.random.normal((128, 3)),
        tf.random.normal((128, 3)),
    )

    test_data = (
        tf.random.normal((64, 3)),
        tf.random.normal((64, 3)),
    )

    trainer = BellmanTrainer(
        model=mock_model,
        net=mock_net,
        train_data=train_data,
        test_data=test_data,
        batch_size=16,
        nu=5,
        lr=1e-3,
    )
    return trainer


# Unit tests
def test_validate_data_valid_tuple():
    """Should accept tuple input of correct structure."""
    data = (tf.zeros((10, 3)), tf.ones((10, 3)))
    assert BellmanTrainer._validate_data(data, "train") == data

def test_validate_data_invalid_type():
    """Should raise TypeError for invalid input."""
    with pytest.raises(TypeError):
        BellmanTrainer._validate_data(tf.random.normal((5, 3)), "train_data")

def test_train_step_returns_loss(mock_trainer):
    """train_step() should return dict containing loss_total."""
    dummy_state = (
        tf.random.normal((16, 3)),
        tf.random.normal((16, 3)),
    )
    loss_dict = mock_trainer.train_step(dummy_state)
    assert "loss_total" in loss_dict
    assert isinstance(loss_dict["loss_total"], tf.Tensor)

def test_evaluate_returns_scalar(mock_trainer):
    """evaluate() should return a numeric scalar."""
    reward = mock_trainer.evaluate(mock_trainer.test_data)["reward_mean"]
    assert isinstance(reward, float)
    assert not np.isnan(reward)


def test_init_shape_mismatch_raises():
    """Should raise ValueError if train/test tensors have mismatched feature dimensions."""
    mock_model = MockModel()
    mock_net = MockNet()

    # Different feature dimensions in first tensor
    train_data = (tf.zeros((100, 3)), tf.ones((100, 3)))
    test_data = (tf.zeros((100, 4)), tf.ones((100, 3))) 

    with pytest.raises(ValueError, match="feature dimensions"):
        BellmanTrainer(mock_model, mock_net, train_data, test_data)

# Integration tests
def test_fit_runs_and_updates_best_reward(mock_trainer):
    """fit() should run end-to-end and update best_reward."""
    mock_trainer.fit(training_steps=5, display_step=2, eval_interval=2)
    # verify training has tracked some evaluations
    assert hasattr(mock_trainer, "eval_lifetime_reward")
    assert len(mock_trainer.eval_lifetime_reward) > 0
    assert isinstance(mock_trainer.best_reward, float)

def test_fit_detects_overfitting(monkeypatch, mock_trainer):
    """Simulate reward stagnation → triggers early stop due to overfitting."""
    # Monkeypatch to force declining rewards
    def declining_reward(*args, **kwargs):
        value = np.random.uniform(-1.0, -0.5)
        return {"reward_mean": value}
    monkeypatch.setattr(mock_trainer, "evaluate", declining_reward)

    mock_trainer.fit(training_steps=10, eval_interval=1, reward_drop_tolerance=2)
    assert mock_trainer.overfitting_detected is True
    assert mock_trainer.overfit_epoch is not None


@pytest.fixture(params=["basic", "debt_free"])
def model_and_net(request):
    
    if request.param == "basic":
        model = BasicModel()
        net = BellmanNet_FOC(model)
    elif request.param == "debt_free":
        model = RiskFree()
        net = BellmanNet_RiskyFree(model)
    else:
        raise ValueError("Unknown model type")

    trainer = BellmanTrainer(model=model, net=net)
    return model, net, trainer


def test_trainer_initialization(model_and_net):
    model, net, trainer = model_and_net

    assert trainer.model is model
    assert trainer.net is net
    assert hasattr(trainer, "train_step"), "Trainer must define train_step method"
    assert hasattr(trainer, "evaluate"), "Trainer must define evaluate method"


def test_trainer_training_step(model_and_net):
    """Test that Trainer can perform a training step without error."""
    model, net, trainer = model_and_net
    
    state = model.sample_state_train(batch_size=4)
    assert isinstance(state, (tuple, list)), "sample_state_train must return a tuple or list"
    assert len(state) in (2, 3), f"Unexpected number of state vars: {len(state)}"

    
    try:
        loss_dict = trainer.train_step(state)
    except Exception as e:
        pytest.fail(f"Trainer.train_step() raised an error: {e}")

    assert isinstance(loss_dict, dict), "Trainer.train_step must return a dict"
    assert "loss_total" in loss_dict, "Returned dict must contain 'loss_total'"


def test_trainer_eval(model_and_net):
    """Test that Trainer can evaluate without error and returns finite metrics."""
    model, net, trainer = model_and_net
    test_data = model.sample_state_test(3)

    try:
        metrics = trainer.evaluate(test_data)
    except Exception as e:
        pytest.fail(f"Trainer.evaluate() raised an error: {e}")

    assert isinstance(metrics, dict), "Trainer.evaluate must return a dict"
    for v in metrics.values():
        assert tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(v))), "Metrics must not contain NaN or inf"