import tensorflow as tf
from typing import Tuple, Union, List, Dict, Any


class EconomicNetwork(tf.keras.Model):
  

    def policy(self, K, B, Z):
        raise NotImplementedError("policy() must be implemented by subclass.")

    def value(self, K, B, Z):
        raise NotImplementedError("value() must be implemented by subclass.")

    def multiplier(self, K, B, Z):
        """Optional: Only used in constrained models (can be empty)."""
        return None


class RiskFree(tf.Module):
    """
    Bellman equation residuals for a company operating under risk-free assumptions.
    """
    
    def __init__(self,
                 phi0: float = 0.01,
                 phi1: float = 0.0,
                 cost_type: str = "None",
                 theta: float = 0.7,
                 delta: float = 0.08,
                 r: float = 0.02,
                 eta0: float = 0.0,
                 eta1: float = 0.07,
                 tau: float = 0.2,
                 s: float = 0.5,
                 rho_z: float = 0.7,
                 std_z: float = 0.15):
        """
        Initialize the economic model parameters.

        Args:
            phi0: Convex term in adjustment cost.
            phi1: Linear term in adjustment cost.
            cost_type: Type of adjustment cost ("None" or "Cost").
            theta: Production function curvature.
            delta: Depreciation rate.
            r: Real interest rate.
            eta0: Fixed effects of external finance cost.
            eta1: Linear component of external finance cost.
            tau: Corporate tax rate.
            s: Fraction of capital liquidatable.
            rho_z: Persistence of productivity shock.
            std_z: Standard deviation of productivity shock.
        """
        super().__init__()

        # Input Validation
        if cost_type not in ["None", "Cost"]:
            raise ValueError(f"Invalid cost_type: {cost_type}. Must be 'None' or 'Cost'.")
        if not (0 <= theta <= 1):
            raise ValueError("Theta must be between 0 and 1.")

        # Adjustment cost parameters 
        self.phi0 = tf.constant(phi0, dtype=tf.float32)
        self.phi1 = tf.constant(phi1, dtype=tf.float32)
        self.cost_mode = cost_type

        # Economic parameters
        self.theta = tf.constant(theta, dtype=tf.float32)
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.r = tf.constant(r, dtype=tf.float32)
        self.beta = 1.0 / (1.0 + self.r)

        # Costly External Finance
        self.eta0 = tf.constant(eta0, dtype=tf.float32)
        self.eta1 = tf.constant(eta1, dtype=tf.float32)

        # Tax and Liquidation
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.s = tf.constant(s, dtype=tf.float32)

        # Shock parameters 
        self.rho_z = tf.constant(rho_z, dtype=tf.float32)
        self.std_z = tf.constant(std_z, dtype=tf.float32)

        # Compute steady state and bounds
        self._compute_steady_state_and_bounds()

    def _compute_steady_state_and_bounds(self):
        """Helper to compute steady state values and grid bounds."""
        self.K_steady = (self.theta / (self.r + self.delta)) ** (1.0 / (1.0 - self.theta))
        self.Pi_steady = self.K_steady ** self.theta
        self.I_steady = self.delta * self.K_steady
        self.E_steady = self.Pi_steady - self.I_steady
        self.V_steady = self.E_steady / self.r
        self.B_steady = (1.0 - self.tau) * tf.pow(self.K_steady, self.theta)

        # Shock bounds (approx +/- 2 std devs)
        term = tf.sqrt(1.0 - self.rho_z**2)
        self.Z_min = tf.exp(-2.0 * self.std_z / term)
        self.Z_max = tf.exp(2.0 * self.std_z / term)

        self.K_min, self.K_max = 0.5 * self.K_steady, 1.5 * self.K_steady
        self.B_min, self.B_max = -0.5 * self.B_steady, 1.0 * self.B_steady

    @tf.function
    def profit(self, K: tf.Tensor, Z: tf.Tensor) -> tf.Tensor:
        """Calculates profit: pi(k, z) = z * k^theta."""
        return Z * tf.pow(K, self.theta)

    @tf.function
    def investment(self, K_prime: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
        """Calculates investment: I = k' - (1-delta)k."""
        return K_prime - (1.0 - self.delta) * K

    @tf.function
    def investment_cost(self, I: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
        """Calculates adjustment cost psi(I, K)."""
        if self.cost_mode == "None":
            return tf.zeros_like(I)
        elif self.cost_mode == "Cost":
            # Avoid division by zero if K is 0 (though unlikely in econ models)
            safe_K = tf.maximum(K, 1e-8) 
            cost = 0.5 * self.phi0 * (tf.pow(I, 2)) / safe_K + self.phi1 * safe_K
            return tf.where(tf.not_equal(I, 0.0), cost, tf.zeros_like(I))
        return tf.zeros_like(I)

    @tf.function
    def collateral_constraint(self, K_prime: tf.Tensor) -> tf.Tensor:
        """Calculates the Right Hand Side (RHS) of the collateral constraint."""
        rhs = (1 - self.tau) * self.Z_min * tf.pow(K_prime, self.theta) \
              + self.tau * self.delta * K_prime + self.s * K_prime
        return rhs

    @tf.function
    def cashflow(self, K: tf.Tensor, K_prime: tf.Tensor, 
                 B: tf.Tensor, B_prime: tf.Tensor, Z: tf.Tensor) -> tf.Tensor:
        """
        Computes current period cash flow.
        e = (1-tau)pi(k,z) - psi(I,K) - I + b'/(1+r(1-tau)) - b
        """
        Pi = self.profit(K, Z)
        I = self.investment(K_prime, K)
        psi = self.investment_cost(I, K)
        
        discount_factor = 1.0 + self.r * (1 - self.tau)
        e = (1 - self.tau) * Pi - psi - I + (B_prime / discount_factor) - B
        return e

    @tf.function
    def external_finance(self, e: tf.Tensor) -> tf.Tensor:
        """Computes cost of external finance based on cash flow sign."""
        indicator = tf.cast(e < 0.0, tf.float32)
        eta = (self.eta0 + self.eta1 * e) * indicator
        return eta

    @tf.function
    def Reward(self, net, K, K_prime, B, B_prime, Z):
        """
        Given current and next states compute current period reward.
        """
        e = self.cashflow(K, K_prime, B, B_prime, Z)
        eta = self.external_finance(e)
        
        # Reward = 股息(负值代表增发) + 融资摩擦成本(通常为负)
        reward = e + eta 
        return reward

    @tf.function
    def lifetime_reward(self, net: EconomicNetwork, cur_state: Tuple[tf.Tensor, ...], T: int):
        """
        Compute the discounted sum of lifetime rewards (Monte Carlo Simulation).
        input: init state (K, B, Z)
        """
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")

        K_t, B_t, Z_t = cur_state
        N = tf.shape(K_t)[0]
        reward_sum = tf.zeros((N, 1), dtype=tf.float32)

        # Time loop
        for t in tf.range(T):
            # 1. The next period state via policy function
            K_next, B_next = net.policy(K_t, B_t, Z_t)
            
            # 2. Calculate current period reward
            e_t = self.cashflow(K_t, K_next, B_t, B_next, Z_t)
            eta_t = self.external_finance(e_t)
            R_t = e_t + eta_t

            # 3. Accumulate discounted rewards
            reward_sum += (self.beta ** tf.cast(t, tf.float32)) * R_t

            # 4. Exogenous shock update (Z state update)
            eps = tf.random.normal((N, 1), mean=0.0, stddev=self.std_z)
            # Log-AR(1) process: ln(Z') = rho * ln(Z) + eps
            Z_next = tf.exp(self.rho_z * tf.math.log(Z_t) + eps)
           
            Z_next = tf.clip_by_value(Z_next, self.Z_min, self.Z_max)

            # 5. State update
            K_t = K_next
            B_t = B_next
            Z_t = Z_next

        return reward_sum

    @tf.function
    def _compute_next_values(self, net: EconomicNetwork, 
                             K_prime: tf.Tensor, B_prime: tf.Tensor, Z: tf.Tensor, 
                             eps: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Helper to compute V(s') and its gradients w.r.t actions.
        Used inside Bellman_residual.
        """
        Z_next = tf.exp(self.rho_z * tf.math.log(Z) + eps)
        Z_next = tf.clip_by_value(Z_next, self.Z_min, self.Z_max)
        
        with tf.GradientTape() as tapeV:
            tapeV.watch([K_prime, B_prime])
            # Assuming net returns [policy_K, policy_B, multiplier, Value]
            # Adjust index 3 based on your actual network architecture
            out = net(tf.concat([K_prime, B_prime, Z_next], axis=1))[:, 3:4]
            
        dVnext_dKp, dVnext_dBp = tapeV.gradient(out, [K_prime, B_prime])
        V_next = out
        
        return V_next, dVnext_dKp, dVnext_dBp

    @tf.function
    def Bellman_residual(self, net: EconomicNetwork, 
                         cur_state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                         nu_K: float = 10.0, 
                         nu_lam: float = 10.0, 
                         nu_FB: float = 1e3) -> Dict[str, tf.Tensor]:
        """
        Computes the loss for the Bellman equation.
        """
        K, B, Z = cur_state
        
        # 1. Current Policy and Value
        K_prime, B_prime = net.policy(K, B, Z)
        Lam = net.multiplier(K, B, Z)
        V_now = net.value(K, B, Z)

        # 2. Current Reward Gradients
        with tf.GradientTape() as tape_R:
            tape_R.watch([K_prime, B_prime])
            e = self.cashflow(K, K_prime, B, B_prime, Z)
            eta = self.external_finance(e)
            R = e + eta
        dR_dKp, dR_dBp = tape_R.gradient(R, [K_prime, B_prime])

        # 3. Future Value Expectations (Monte Carlo with 2 shocks)
        eps1 = tf.random.normal(tf.shape(Z), mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal(tf.shape(Z), mean=0.0, stddev=self.std_z)

        V_next1, dVnext_dKp1, dVnext_dBp1 = self._compute_next_values(net, K_prime, B_prime, Z, eps1)
        V_next2, dVnext_dKp2, dVnext_dBp2 = self._compute_next_values(net, K_prime, B_prime, Z, eps2)

        # 4. Compute Residuals
        # Value Function Residual
        res_V_1 = V_now - (R + self.beta * V_next1)
        res_V_2 = V_now - (R + self.beta * V_next2)
        
        # Euler Equation (Capital) Residual
        res_FOC_K_1 = dR_dKp + self.beta * dVnext_dKp1
        res_FOC_K_2 = dR_dKp + self.beta * dVnext_dKp2

        # Multiplier Residual
        res_FOC_lam_1 = self._lam_residual(Lam, dR_dBp, dVnext_dBp1)
        res_FOC_lam_2 = self._lam_residual(Lam, dR_dBp, dVnext_dBp2)

        # Fisher-Burmeister Residual (Constraints)
        res_FB = self._fb_residual(K_prime, B_prime, Lam)

        # 5. Combine Losses (Product of residuals approach)
        residual_V = res_V_1 * res_V_2
        residual_FOC_K = res_FOC_K_1 * res_FOC_K_2
        residual_FOC_lam = res_FOC_lam_1 * res_FOC_lam_2

        loss_total = tf.reduce_mean(
            residual_V + 
            nu_K * residual_FOC_K + 
            nu_lam * residual_FOC_lam + 
            nu_FB * res_FB
        )

        return {
            "loss_total": loss_total,
            "loss_V": tf.reduce_mean(tf.abs(residual_V)),
            "loss_FOC_K": tf.reduce_mean(tf.abs(residual_FOC_K)),
            "loss_FB": tf.reduce_mean(tf.abs(res_FB)),
        }

    def _fb_residual(self, K_prime, B_prime, Lam):
        """Fisher-Burmeister residual for complementarity conditions."""
        g = self.collateral_constraint(K_prime) - B_prime
        # FB(a,b) = a + b - sqrt(a^2 + b^2) = 0 <=> a>=0, b>=0, ab=0
        fb_val = g + Lam - tf.sqrt(tf.square(g) + tf.square(Lam) + 1e-8) # Added epsilon for stability
        return tf.square(fb_val)

    def _lam_residual(self, lam, dR_dBP, dVnext_dBP):
        """Target residual for Lambda based on FOC w.r.t Debt."""
        lam_target = - (dR_dBP + self.beta * dVnext_dBP)
        return lam - lam_target

    def sample_state_train(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Uniformly samples states (K, B, Z) for training.
        """
        K_sample = tf.random.uniform(
            shape=(batch_size, 1), minval=self.K_min, maxval=self.K_max, dtype=tf.float32
        )

        Z_sample = tf.random.uniform(
            shape=(batch_size, 1), minval=self.Z_min, maxval=self.Z_max, dtype=tf.float32
        )

        B_sample = tf.random.uniform(
            shape=(batch_size, 1), minval=self.B_min, maxval=self.B_max, dtype=tf.float32
        )
        
        # tf.print("Initialized", batch_size, "state samples (K, B, Z)")
        return (K_sample, B_sample, Z_sample)

    def sample_state_test(self, test_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Samples states (K, B, Z) around the steady state for testing.
        """
      
        K_eval = self.K_steady * (1.0 + tf.random.normal(
            (test_size, 1), mean=0.0, stddev=0.1
        ))
        K_eval = tf.clip_by_value(K_eval, self.K_min, self.K_max)

     
        B_eval = (1.0 + tf.random.normal(
            (test_size, 1), mean=0.0, stddev=0.5
        ))
        B_eval = tf.clip_by_value(B_eval, self.B_min, self.B_max)
    
    
        Z_eval = tf.exp(tf.random.normal(
            (test_size, 1), mean=0.0, stddev=self.std_z
        ))
        Z_eval = tf.clip_by_value(Z_eval, self.Z_min, self.Z_max)

        return (K_eval, B_eval, Z_eval)



class RiskDebt(RiskFree):
    """
    Economic model extending RiskFree to include risky debt and default possibilities.
    """

    def __init__(self, prev_net: EconomicNetwork, alpha: float = 0.0, **kwargs):
        """
        Args:
            prev_net: The trained network from the previous iteration (for value function approximation).
            alpha: Bankruptcy cost parameter.
            **kwargs: Arguments passed to RiskFree.
        """
        super().__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.prev_net = prev_net

    @tf.function
    def recovery_value(self, K: tf.Tensor, Z: tf.Tensor) -> tf.Tensor:
        """Calculates value recovered by creditors in case of default."""
        profit = self.profit(K, Z)
        val = (1 - self.alpha) * ((1 - self.tau) * profit + (1 - self.delta) * K)
        return val

        
    @tf.function(reduce_retracing=True)
    def get_r_tilde(self, net: EconomicNetwork, Kp: tf.Tensor, Bp: tf.Tensor, Z: tf.Tensor, N_mc: int = 100) -> tf.Tensor:
        """
        Computes the risky interest rate r_tilde using Monte Carlo integration.
        
        Returns a high penalty rate if default probability is 100%.
        """
        
        eps = tf.random.normal((N_mc,), mean=0.0, stddev= self.std_z)
        logZ_next = self.rho_z * tf.math.log(Z) + eps             # (N, N_mc)
        Z_next = tf.exp(logZ_next)                            # (N, N_mc)

        N = tf.shape(Kp)[0]
        Kp_expand = tf.repeat(Kp, repeats=N_mc, axis=1)          # (N, N_mc)
        bp_expand = tf.repeat(Bp, repeats=N_mc, axis=1)          # (N, N_mc)


        Kp_flat = tf.reshape(Kp_expand, (-1, 1))                 # (N*N_mc, 1)
        bp_flat = tf.reshape(bp_expand, (-1, 1))                 # (N*N_mc, 1)
        Z_flat  = tf.reshape(Z_next, (-1, 1))                    # (N*N_mc, 1)
        
        V_next = net.value(Kp_flat, bp_flat, Z_flat)
        R_values = self.recovery_value(Kp_flat, Z_flat)

        V_next = tf.reshape(V_next, (N, N_mc))
        R_values = tf.reshape(R_values, (N, N_mc))

        default_mask = tf.cast(V_next <= 0.0, tf.float32)     # (N, N_mc), default set

        P_default = tf.reduce_mean(default_mask, axis=1, keepdims=True)  # (N,1)
        R_mean = tf.reduce_mean(R_values * default_mask, axis=1, keepdims=True)  # (N,1)

        eps_clip = 1e-8
        P_safe = tf.clip_by_value(P_default, 0.0, 1.0 - eps_clip)

        r_tilde = ((1.0 + self.r) - R_mean / Bp) / (1.0 - P_safe) - 1.0
        # For states with certain default, set a very high r_tilde (equivalent to no bond issuance)
        r_tilde = tf.where(P_default >= 1.0 - 1e-6, tf.constant(1e8, dtype=tf.float32), r_tilde)
        return tf.stop_gradient(r_tilde)

    def mask_default_set(self, net, Kp, Bp, Z):
        r_tilde = self.get_r_tilde(net, Kp, Bp, Z)
        mask_default = tf.cast(r_tilde >= 1e7, tf.float32)
        return mask_default


    def get_r_tilde_test(self, net, test_data):
        K, B, Z = test_data
        Kp, Bp = net.policy(K, B, Z)
        r_tilde = self.get_r_tilde(net, Kp, Bp, Z)
        return r_tilde

    
    @tf.function
    def mask_default_set(self, net, Kp, Bp, Z):
        r_tilde = self.get_r_tilde(net, Kp, Bp, Z)
        return tf.cast(r_tilde >= 1e7, tf.float32)

    def get_r_tilde_test(self, net, test_data):
        K, B, Z = test_data
        Kp, Bp = net.policy(K, B, Z)
        return self.get_r_tilde(net, Kp, Bp, Z)

    @tf.function
    def Bellman_residual(self, net, cur_state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], nu: float = 10.0) -> Dict[str, tf.Tensor]:
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("cur_state must be a tuple or list (K, B, Z)")

        K, B, Z = cur_state
        self.nu = nu

        Kp, Bp = net.policy(K, B, Z)
        V_now = net.value(K, B, Z)

        with tf.GradientTape() as tape_R:
            tape_R.watch([Kp, Bp])
            e = self.cashflow(K, Kp, B, Bp, Z)
            eta = self.external_finance(e)
            R = e + eta
        dR_dKp, dR_dBp = tape_R.gradient(R, [Kp, Bp])
        del tape_R

        eps1 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)

        def next_values(eps):
            mask_default = self.mask_default_set(self.prev_net, Kp, Bp, Z)
            K_restart = tf.ones_like(K) * self.K_steady
            B_restart = tf.zeros_like(B)
            Kp_eff = Kp * (1 - mask_default) + K_restart * mask_default
            Bp_eff = Bp * (1 - mask_default) + B_restart * mask_default

            Z_next = tf.exp(self.rho_z * tf.math.log(Z) + eps)
            Z_next = tf.clip_by_value(Z_next, self.Z_min, self.Z_max)

            V_next = net.value(Kp_eff, Bp_eff, Z_next)
            with tf.GradientTape() as tapeV:
                tapeV.watch([Kp_eff, Bp_eff])
                out = net(tf.concat([Kp_eff, Bp_eff, Z_next], axis=1))[:, 2:3]
                dVnext_dKp, dVnext_dBp = tapeV.gradient(out, [Kp_eff, Bp_eff])
            return V_next, dVnext_dKp, dVnext_dBp

        V_next1, dVnext_dKp1, dVnext_dBp1 = next_values(eps1)
        V_next2, dVnext_dKp2, dVnext_dBp2 = next_values(eps2)

        res_V1 = V_now - (R + self.beta * V_next1)
        res_V2 = V_now - (R + self.beta * V_next2)
        res_FOC_K1 = dR_dKp + self.beta * dVnext_dKp1
        res_FOC_K2 = dR_dKp + self.beta * dVnext_dKp2
        res_FOC_B1 = dR_dBp + self.beta * dVnext_dBp1
        res_FOC_B2 = dR_dBp + self.beta * dVnext_dBp2

        residual_V = res_V1 * res_V2
        residual_FOC_K = res_FOC_K1 * res_FOC_K2
        residual_FOC_B = res_FOC_B1 * res_FOC_B2

        loss_total = tf.reduce_mean(residual_V + nu * residual_FOC_K + nu * residual_FOC_B)
        return {
            "loss_total": loss_total,
            "loss_V": tf.reduce_mean(tf.abs(residual_V)),
            "loss_FOC_K": tf.reduce_mean(tf.abs(residual_FOC_K)),
            "loss_FOC_B": tf.reduce_mean(tf.abs(residual_FOC_B)),
        }
