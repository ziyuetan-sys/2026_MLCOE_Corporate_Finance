import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Dict, Union, Any

class BasicModel(tf.Module):
    """
    A TensorFlow-based economic model representing a firm with capital adjustment costs 
    and stochastic productivity shocks.

    Attributes:
        phi0 (tf.Tensor): Convex term parameter in the adjustment cost function.
        phi1 (tf.Tensor): Linear term parameter in the adjustment cost function.
        cost_mode (str): The type of adjustment cost ('None' or 'Cost').
        theta (tf.Tensor): Production function curvature parameter (0 < theta < 1).
        delta (tf.Tensor): Depreciation rate (0 < delta < 1).
        r (tf.Tensor): Real interest rate.
        rho_z (tf.Tensor): Persistence of the productivity shock z.
        std_z (tf.Tensor): Standard deviation of the productivity shock error term.
    """
    def __init__(self,
        phi0 = 0.01,      
        phi1 = 0.0,           
        cost_type = "None",
        theta=0.7,         
        delta=0.08,       
        r=0.04,          
        rho_z=0.7,        
        std_z=0.15,       
        ):
        """
        Initialize the economic model parameters and compute steady-state values.
        
        Args:
            phi0: Quadratic adjustment cost parameter.
            phi1: Linear adjustment cost parameter.
            cost_type: Configuration for cost function ('None' or 'Cost').
            theta: Production elasticity of capital.
            delta: Capital depreciation rate.
            r: Risk-free interest rate.
            rho_z: Autoregressive coefficient for shock process.
            std_z: Standard deviation of shock innovation.
        """
        super().__init__()
        
        # Input Validation
        if cost_type not in ["None", "Cost"]:
            raise ValueError(f"Invalid cost_type: {cost_type}. Must be 'None' or 'Cost'.")
        if not (0 < theta < 1):
            raise ValueError("Theta must be between 0 and 1.")
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1.")
        
        # adjustment cost parameters 
        self.phi0 = tf.constant(phi0, dtype = tf.float32)
        self.phi1 = tf.constant(phi1,dtype = tf.float32)
        self.cost_mode = cost_type

        # economic parameters
        self.theta = tf.constant(theta, dtype=tf.float32)
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.r = tf.constant(r, dtype=tf.float32)
        self.beta = 1.0 / (1.0 + self.r)

        # shock parameters 
        self.rho_z = tf.constant(rho_z, dtype=tf.float32)
        self.std_z = tf.constant(std_z, dtype=tf.float32)

        # Compute steady state
        self.K_steady = (self.theta / (self.r + self.delta)) ** (1.0 / (1.0 - self.theta)) # Capital
        self.Pi_steady = self.K_steady ** self.theta # Profit 
        self.I_steady = self.delta * self.K_steady # Investment
        self.E_steady = self.Pi_steady - self.I_steady # Cash flow
        self.V_steady = self.E_steady / self.r # Company Value      

        self.Z_min = tf.exp(-2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))
        self.Z_max = tf.exp(2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))

        self.K_min, self.K_max = 0.5 * self.K_steady, 1.5 * self.K_steady
  
    
    # Model function
    @tf.function
    def profit(self, K: tf.Tensor, Z: tf.Tensor) -> tf.Tensor:
        K = tf.reshape(K, (-1, 1))
        Z = tf.reshape(Z, (-1, 1))
        """Computes production profit: pi(k, z) = z * k^theta."""
        return Z * tf.pow(K, self.theta)

   
    @tf.function
    def investment(self, K_prime: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
        """Computes investment required to reach K_prime from K: I = K' - (1-delta)K."""
        K_prime = tf.reshape(K_prime, (-1, 1))
        K = tf.reshape(K, (-1, 1))
        return K_prime - (1.0 - self.delta) * K

    @tf.function
    def investment_cost(self, I: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
        """
        Computes the adjustment cost psi(I, K).
        
        Returns:
            tf.Tensor: The calculated cost.
        """
        I = tf.reshape(I, (-1, 1))
        K = tf.reshape(K, (-1, 1))
        
        if self.cost_mode == "None":
            # Return zeros maintaining the shape of input I
            return tf.zeros_like(I) * (I + K)
        elif self.cost_mode == "Cost":
            # Quadratic adjustment cost
            # Note: tf.where handles the case where I=0 to avoid division by zero issues if applicable,
            # though here the formula is safe if K > 0.
            return tf.where(
                tf.not_equal(I, 0.0),
                0.5 * self.phi0 * (tf.pow(I, 2)) / K + self.phi1 * K,
                tf.zeros_like(I) * (I + K)
            )
        else:
            raise ValueError("cost_type must be 'None' or 'Cost'.")
        
    @tf.function
    def cashflow(self, K: tf.Tensor, Z: tf.Tensor, K_prime: tf.Tensor) -> tf.Tensor:
        """Computes current period utility/cashflow: Profit - Cost - Investment."""
        K = tf.reshape(K, (-1, 1))
        Z = tf.reshape(Z, (-1, 1))
        K_prime = tf.reshape(K_prime, (-1, 1))
        Pi = self.profit(K, Z)
        I = self.investment(K_prime, K)
        psi = self.investment_cost(I, K)
        return Pi - psi - I

        
   
    @tf.function
    def lifetime_reward(self, net: Any, cur_state: Tuple[tf.Tensor, tf.Tensor], T: int) -> tf.Tensor:
        """
        Simulates the trajectory for T periods and computes discounted sum of rewards.
        
        Args:
            net: A neural network object with a call method returning next state.
            cur_state: Tuple (K_t, Z_t).
            T: Number of periods to simulate.
        """
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")
        
        K_t, Z_t = cur_state
      
        K_t = tf.reshape(K_t, (-1, 1))
        Z_t = tf.reshape(Z_t, (-1, 1))
        
        K_steady = self.K_steady
        N = tf.shape(K_t)[0]
        reward_sum = tf.zeros((N,1), dtype=tf.float32)


        for t in tf.range(T):
            # Next period K‘, B‘
            out = net(tf.concat([K_t, Z_t], axis=1))
            K_next = out[:, :1]
           
            # Current Reward
            R_t = self.cashflow(K = K_t, Z = Z_t, K_prime = K_next)

            reward_sum += (self.beta ** tf.cast(t, tf.float32)) * R_t

            # shock 
            eps = tf.random.normal((N, 1), mean=0.0, stddev=self.std_z)
            Z_t = tf.exp(self.rho_z * tf.math.log(Z_t) + eps)

            # State update
            K_t = K_next
        return reward_sum


    @tf.function
    def Euler_residuals(self, net: Any, cur_state: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Computes the Euler Equation residual using the 'All-in-One' (double sampling) method.

        The Euler equation represents the intertemporal optimality condition:
        1 + d(Psi)/dI = beta * E_t [ MPK_{t+1} - d(Psi')/dK' + (1-delta)*(1 + d(Psi')/dI') ]

        Args:
            net: Neural network object with a .policy() method.
            cur_state: Tuple (K_t, Z_t).

        Returns:
            tf.Tensor: A scalar representing the mean squared residual.
        """
        K_t, Z_t = cur_state

        N = K_t.shape[0]
       

        # --- current period terms ---
        K_prime = net.policy(K_t, Z_t)
        I_now = self.investment(K_prime, K_t)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([I_now, K_t])
            psi_now = self.investment_cost(I_now, K_t)
        psi_I_now = tape.gradient(psi_now, I_now)
        del tape
        RHS = 1.0 + psi_I_now

        #  next-period term 
        def euler_term(eps):
            Z_next = tf.exp(self.rho_z * tf.math.log(Z_t) + eps)
            K_next = tf.identity(K_prime)
            K_next_prime = net.policy(K_next, Z_next)
            I_next = self.investment(K_next_prime, K_next)
            pi_k_next = self.theta * Z_next * tf.pow(K_next, self.theta - 1)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch([I_next, K_next])
                psi_next = self.investment_cost(I_next, K_next)
            psi_I_next = t2.gradient(psi_next, I_next)
            psi_K_next = t2.gradient(psi_next, K_next)
            del t2
            return pi_k_next - psi_K_next + (1 - self.delta) * (1 + psi_I_next)

        # AiO 
        eps1 = tf.random.normal((N, 1), mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal((N, 1), mean=0.0, stddev=self.std_z)
        term1, term2 = euler_term(eps1), euler_term(eps2)
        LHS1 = self.beta * term1
        LHS2 = self.beta * term2
        euler_resid = tf.reduce_mean((LHS1 - RHS) * (LHS2 - RHS))

        return euler_resid

  
    @tf.function
    def Bellman_residual(self, net: Any, cur_state: Tuple[tf.Tensor, tf.Tensor], nu: float = 10.0) -> Dict[str, tf.Tensor]:
        """
        Compute combined residuals: o
            Loss(theta) = E[ residual_value^2 + nu residual_foc^2 ]
            FOC = d rewward/d K_prime + 1/(1 + r) * d E_eps[dV_prime / d K_prime]
        """
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")
        K, Z = cur_state
        self.nu = nu
       
       
        K_prime = net.policy(K, Z)
        V_now = net.value(K, Z)

        with tf.GradientTape(persistent=True) as tape_R:
            tape_R.watch(K_prime)
            R = self.cashflow(K, Z, K_prime)
        dr_dKprime = tape_R.gradient(R, K_prime)
        del tape_R

        # Draw two independent shocks eps1, eps2
        eps1 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)

        # Define helper to compute next-period stuff under different eps 
        def next_values(eps):
            Z_next = tf.exp(self.rho_z * tf.math.log(Z) + eps)
            Z_next = tf.clip_by_value(Z_next, self.Z_min, self.Z_max)
            val_next = net(tf.concat([K_prime,Z_next], axis=1))
            V_next = val_next[:, :1] 
           
            with tf.GradientTape() as tapeV:
                tapeV.watch(K_prime)
                V_grad = net(tf.concat([K_prime, Z_next], axis=1))[:, :1]
            dVnext_dKp = tapeV.gradient(V_grad, K_prime)
            return V_next, dVnext_dKp
        

        V_next1, dVnext_dKp1 = next_values(eps1)
        V_next2, dVnext_dKp2 = next_values(eps2)

       
        # Residuals under eps1, eps2 respectively
        residual_V_eps1 = V_now - (R + self.beta * V_next1)
        residual_V_eps2 = V_now - (R + self.beta * V_next2)

        residual_FOC_eps1 = dr_dKprime + self.beta * dVnext_dKp1
        residual_FOC_eps2 = dr_dKprime + self.beta * dVnext_dKp2


        residual_V = residual_V_eps1 * residual_V_eps2
        residual_FOC = residual_FOC_eps1 * residual_FOC_eps2

        # Total loss 
        loss = tf.reduce_mean(residual_V + self.nu * residual_FOC)

        loss_V = tf.reduce_mean(tf.abs(residual_V))
        loss_FOC = tf.reduce_mean(tf.abs(residual_FOC))

        loss_dict = {
            "loss_total": loss,
            "loss_V": loss_V,
            "loss_FOC": loss_FOC,
           
        }

        return loss_dict
    

    def sample_state_train(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples random states from uniform distribution for training."""
        K_sample = tf.random.uniform((batch_size, 1), minval=self.K_min, maxval=self.K_max, dtype=tf.float32)
        Z_sample = tf.random.uniform((batch_size, 1), minval=self.Z_min, maxval=self.Z_max, dtype=tf.float32)
        return (K_sample, Z_sample)

    def sample_state_test(self, test_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples states near steady state for testing/evaluation."""
        K_eval = tf.random.uniform((test_size, 1), minval=self.K_min, maxval=self.K_max, dtype=tf.float32)
      
        Z_eval = tf.exp(tf.random.normal((test_size, 1), mean=0.0, stddev=self.std_z))
        Z_eval = tf.clip_by_value(Z_eval, self.Z_min, self.Z_max)
        return (K_eval, Z_eval)

