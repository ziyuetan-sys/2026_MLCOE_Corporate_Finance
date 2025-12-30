import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm
import numpy as np

class BellmanTrainer:
    def __init__(self,
                 model,
                 net, 
                 train_data = None, # tuple or list
                 test_data = None, # tuple or list
                 nu = 10,
                 batch_size=64,
                 optimizer = tf.keras.optimizers.Adam,
                 # loss = "bellman"
                 hidden_dim = [64, 64],
                 activation='relu', 
                 mode='split',
                 lr=1e-3):
        self.model = model
        self.net =  net
        self.nu = nu
        self.batch_size = batch_size
        self.opt = optimizer(learning_rate=lr, clipnorm=5.0)
        self.lr = lr
        self.test_data = test_data
        self.train_data = train_data 

        # Validate train/test data
        if train_data is not None:
            if not (isinstance(train_data, (tuple, list)) and len(train_data) >= 2):
                    raise TypeError("train_data must be a tuple/list of (K, [B], Z) tensors")
        if test_data is not None:
            if not (isinstance(test_data, (tuple, list)) and len(test_data) >= 2):
                    raise TypeError("test_data must be a tuple/list of (K, [B], Z) tensors")
        if train_data is not None and test_data is not None:
            if len(train_data) != len(test_data):
                raise ValueError(
                    f"train_data and test_data must have the same length, "
                    f"got {len(train_data)} and {len(test_data)}"
                )

    
    @tf.function
    def train_step(self, state_batch):
        with tf.GradientTape() as tape:
            loss_dict = self.model.Bellman_residual( # 加上用其他loss的option
            self.net, state_batch, self.nu
        )
        loss = loss_dict["loss_total"]
            
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
        return  loss_dict
    

    def fit(self,
            training_steps = 10000, 
            display_step=100, 
            eval = True, # Evaluate during training
            eval_interval= 10,
            early_stop = True, #in case of overfitting
            reward_drop_tolerance = 10,
            n_eval_points = 8192):
        
        self.training_steps =  training_steps
            
        # Test data
        if eval:
            #self.eval_epochs, self.eval_euler_resid, self.eval_lifetime_reward = [], [], []
            self.eval_epochs,  self.eval_lifetime_reward = [], []
            if (self.test_data is None):
                # generate test data
                self.test_data = self.model.sample_state_test(test_size=n_eval_points)   
        else:
            #self.eval_epochs, self.eval_euler_resid, self.eval_lifetime_reward = None, None, None
            self.eval_epochs,  self.eval_lifetime_reward = None, None

        # Test for whether overfitting
        self.best_reward = -np.inf
        self.best_epoch,self.best_weights,  self.overfit_epoch   = None, None, None
        self.overfitting_detected = False
        reward_decline_counter = 0 

        # Training data
        if  (self.train_data is not None):
            train_K =  self.train_data[0]
            N_train = tf.shape(train_K)[0]
            
        
        progress_bar = tqdm(range(1, training_steps + 1),
                            desc="Training progress",
                            ncols=120,
                            leave=True,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')


       
        tf.print("Starting training ...")
        # Initialize network weights
        self.net.initialize_weights()
        tf.print(" Network weights initialized.")


        for epoch in progress_bar:

            if self.train_data is not None:
                # Sample mini-batch from given data
                idx = tf.random.shuffle(tf.range(N_train))[:self.batch_size]
                state_batch = tuple(tf.gather(x, idx) for x in self.train_data)

            else:
                # Random generate training data from state space
                state_batch =  self.model.sample_state_train(self.batch_size)

            loss_dict  = self.train_step(state_batch)
            loss_float_dict = {k: float(v) for k, v in loss_dict.items() if v is not None}

            for k, v in loss_float_dict.items():
                if not hasattr(self, f"{k}_list"):
                    setattr(self, f"{k}_list", [])
                getattr(self, f"{k}_list").append(v)

         
            progress_bar.set_postfix({
                k: f"{v:.4e}" for k, v in loss_float_dict.items()
            })
            

            # add an evaluate mode
            if eval and (epoch % eval_interval == 0 or epoch == 1):
                #euler_resid, lifetime_reward = self.evaluate(self.test_data)
                lifetime_reward = self.evaluate(self.test_data)
               
                self.eval_epochs.append(epoch)
                #self.eval_euler_resid.append(euler_resid)
                self.eval_lifetime_reward.append(lifetime_reward)
            
                # Test overfitting
                if lifetime_reward > self.best_reward:
                        # update best 
                        self.best_reward = lifetime_reward
                        self.best_epoch = epoch
                        self.best_weights = self.net.get_weights()
                        reward_decline_counter = 0
                else:
                        reward_decline_counter += 1
                if (not self.overfitting_detected) and  (reward_decline_counter >=  reward_drop_tolerance):
                    self.overfitting_detected = True
                    self.overfit_epoch = epoch

                    tf.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!! Overfitting detected at epoch {epoch}. !!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        f"Best reward = {self.best_reward:.4e} (epoch {self.best_epoch})")
                
                    if early_stop:
                        self.net.set_weights(self.best_weights)
                        break
            else:
                #euler_resid, lifetime_reward = None, None
                lifetime_reward = None


            if epoch % display_step == 0 or epoch == 1:
                loss_str = " | ".join([f"{k}={v:.6e}" for k, v in loss_float_dict.items()])
                tf.print(f"Epoch {epoch:>5} | {loss_str}")

                if eval and ( lifetime_reward  is not None):
                    tf.print(
                        f"★ Eval @ epoch {epoch}: "
                        #f"mean Euler resid = {euler_resid:.3e}; "
                        f"lifetime reward = {lifetime_reward:.3e}"
                    )
        if eval:
            if self.overfitting_detected:
                tf.print(f"Training stopped early (overfitting at epoch {self.overfit_epoch}). "
                            f"Restored best model from epoch {self.best_epoch}.")
                self.net.set_weights(self.best_weights)
            else:
                tf.print("Training completed — no overfitting detected.")
            tf.print(f" Best reward: {self.best_reward:.3e} @ epoch {self.best_epoch}")
    # 给每个model定义不同的evaluate senario
    def evaluate(self, test_data):
    
        reward = tf.stop_gradient(
            self.model.lifetime_reward(self.net, test_data, T=200)
        )

        mean_reward = tf.reduce_mean(reward)
        return float(mean_reward)
