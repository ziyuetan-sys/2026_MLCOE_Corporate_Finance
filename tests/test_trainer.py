import pytest
import tensorflow as tf
import numpy as np
from dynamic_model.networks import BellmanNet_FOC
from dynamic_model.trainer import BellmanTrainer  
from dynamic_model.Basic import BasicModel




@pytest.fixture
def trainer():
    model = BasicModel()
    net = BellmanNet_FOC(model=model, hidden_dim=[8, 8])
    train_data = model.sample_state_train(128)
    return BellmanTrainer(model=model, net=net, train_data=train_data)

def test_trainer_init_checks(trainer):
    assert isinstance(trainer.net, tf.keras.Model)
    assert hasattr(trainer.model, "Bellman_residual")
    assert trainer.batch_size == 64

def test_train_step_runs(trainer):
    batch = trainer.model.sample_state_train(trainer.batch_size)
    loss_dict = trainer.train_step(batch)
    assert "loss_total" in loss_dict
    assert isinstance(loss_dict["loss_total"], tf.Tensor)
    assert not tf.math.is_nan(loss_dict["loss_total"])


def test_evaluate_runs(trainer):
    test_data = trainer.model.sample_state_test(20)
    val = trainer.evaluate(test_data)
    assert isinstance(val, float)

def test_fit_runs_short(trainer):
    trainer.fit(training_steps=5, display_step=1, eval=False)
    tracked_losses = [k for k in trainer.__dict__.keys() if k.startswith("loss_")]
    assert len(tracked_losses) >= 1, "Trainer should have loss listes"
    print("Tracked losses:", tracked_losses)
   
    if hasattr(trainer, "loss_total_list"):
        assert all(isinstance(v, float) for v in trainer.loss_total_list[:3]), "loss_total_list should store floats."