import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
import numpy as np


class BellmanTrainer:
    """
    A trainer class to optimize Bellman residual-based models.

    This class is designed for clarity and maintainability:
    - Explicit data validation
    - Modular structure
    - Docstrings for all public methods
    """

    def __init__(
        self,
        model: Model,
        net: Model,
        train_data=None,
        test_data=None,
        nu: int = 10,
        batch_size: int = 64,
        optimizer=tf.keras.optimizers.Adam,
        hidden_dim=None,
        activation: str = "relu",
        mode: str = "split",
        lr: float = 1e-3,
    ):
        """
        Initialize the BellmanTrainer.

        Args:
            model: A model object implementing `Bellman_residual` and `lifetime_reward`.
            net: Neural network to be trained.
            train_data: Tuple or list of training tensors (e.g., (K, [B], Z)).
            test_data: Tuple or list of test tensors (same structure as train_data).
            nu: Scaling factor for Bellman residual loss.
            batch_size: Mini-batch size for training.
            optimizer: Optimizer class from tf.keras.optimizers.
            hidden_dim: Hidden layer sizes (optional, used externally).
            activation: Activation function.
            mode: Operating mode (e.g., "split").
            lr: Learning rate.
        """
        self.model = model
        self.net = net
        self.nu = nu
        self.batch_size = batch_size
        self.lr = lr
        self.opt = optimizer(learning_rate=lr, clipnorm=5.0)
        self.hidden_dim = hidden_dim or [64, 64]
        self.activation = activation
        self.mode = mode

        self.train_data = self._validate_data(train_data, "train_data")
        self.test_data = self._validate_data(test_data, "test_data")

        if train_data is not None and test_data is not None:
            if len(train_data) != len(test_data):
                raise ValueError(
                    f"train_data and test_data must have the same length, "
                    f"got {len(train_data)} and {len(test_data)}"
                )

    @staticmethod
    def _validate_data(data, name: str):
        """Validate that input data is a tuple/list of tensors with sufficient length."""
        if data is None:
            return None
        if not (isinstance(data, (tuple, list)) and len(data) >= 2):
            raise TypeError(f"{name} must be a tuple/list of (K, [B], Z) tensors")
        return data

    @tf.function
    def train_step(self, state_batch):
        """Compute gradients and apply one optimization step."""
        with tf.GradientTape() as tape:
            loss_dict = self.model.Bellman_residual(self.net, state_batch, self.nu)
            loss = loss_dict["loss_total"]

        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
        return loss_dict

    def fit(
        self,
        training_steps: int = 10_000,
        display_step: int = 100,
        eval: bool = True,
        eval_interval: int = 10,
        early_stop: bool = True,
        reward_drop_tolerance: int = 10,
        n_eval_points: int = 8192,
    ):
        """Train the model using Bellman residual minimization."""
        if self.train_data is not None:
            train_K = self.train_data[0]
            N_train = tf.shape(train_K)[0]

        # Setup test data
        if eval and self.test_data is None:
            self.test_data = self.model.sample_state_test(test_size=n_eval_points)

        self.eval_epochs, self.eval_lifetime_reward = [], []
        self.best_reward = -np.inf
        self.best_epoch = self.best_weights = self.overfit_epoch = None
        self.overfitting_detected = False
        reward_decline_counter = 0

        progress = tqdm(
            range(1, training_steps + 1),
            desc="Training progress",
            ncols=120,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        tf.print("Starting training ...")
        self.net.initialize_weights()
        tf.print("Network weights initialized.")

        for epoch in progress:
            # Prepare batch
            if self.train_data is not None:
                idx = tf.random.shuffle(tf.range(N_train))[:self.batch_size]
                state_batch = tuple(tf.gather(x, idx) for x in self.train_data)
            else:
                state_batch = self.model.sample_state_train(self.batch_size)

            loss_dict = self.train_step(state_batch)
            loss_float_dict = {k: float(v) for k, v in loss_dict.items() if v is not None}
            for k, v in loss_float_dict.items():
                getattr(self, f"{k}_list", []).append(v) if hasattr(self, f"{k}_list") else setattr(self, f"{k}_list", [v])

            progress.set_postfix({k: f"{v:.4e}" for k, v in loss_float_dict.items()})

            
            # ---- Periodic evaluation ----
            lifetime_reward = None
            if eval and (epoch % eval_interval == 0 or epoch == 1):
                lifetime_reward = self.evaluate(self.test_data)
                self.eval_epochs.append(epoch)
                self.eval_lifetime_reward.append(lifetime_reward)

                # Update best metrics
                if lifetime_reward > self.best_reward:
                    self.best_reward = lifetime_reward
                    self.best_epoch = epoch
                    self.best_weights = self.net.get_weights()
                    reward_decline_counter = 0
                else:
                    reward_decline_counter += 1

                # Overfitting detection
                if not self.overfitting_detected and reward_decline_counter >= reward_drop_tolerance:
                    self.overfitting_detected = True
                    self.overfit_epoch = epoch

                    if early_stop:
                        # Only print + stop when early_stop=True
                        tf.print(
                            f"Overfitting detected at epoch {epoch}. "
                            f"Best reward = {self.best_reward:.4e} (epoch {self.best_epoch})"
                        )
                        self.net.set_weights(self.best_weights)
                        break

            # ---- Display progress ----
            if epoch % display_step == 0 or epoch == 1:
                loss_str = " | ".join([f"{k}={v:.6e}" for k, v in loss_float_dict.items()])
                tf.print(f"Epoch {epoch:>5} | {loss_str}")
                if eval and lifetime_reward is not None:
                    tf.print(f" Eval @ epoch {epoch}: lifetime reward = {lifetime_reward:.3e}")

        # Training complete
        tf.print("Training completed.")
        if eval:
            if self.overfitting_detected:
                tf.print(f"Best reward: {self.best_reward:.3e} @ epoch {self.best_epoch}")
                if early_stop:
                    tf.print(
                        f"Training stopped early (overfitting at epoch {self.overfit_epoch}). "
                        f"Restored best model from epoch {self.best_epoch}."
                    )
                    self.net.set_weights(self.best_weights)
               

    def evaluate(self, test_data):
        """Evaluate model lifetime reward on test data."""
        reward = tf.stop_gradient(
            self.model.lifetime_reward(self.net, test_data, T=200)
        )
        return float(tf.reduce_mean(reward))