import tensorflow as tf
from tensorflow.keras import layers, Model
# BasicModel
# RiskFree
# RiskDebt

class BasicModel(tf.Module):
    def __init__(self,
        phi0 = 0.01,       # Convex term in the adjustment cost function
        phi1 = 0.0,            # Linear term in the adjustment cost function
        cost_type = "None",
        theta=0.7,         # The production function curvature (\pi(k,z) = z*k^θ)
        delta=0.08,        # Depreciation rate: fraction of capital lost each period
        r=0.04,            # Real interest rate: discounting future returns
        rho_z=0.7,         # The persistence of z shock
        std_z=0.15,        # The standard deviation of the error term 
        ):
        super().__init__()

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
    # profit(k,z) = z * k^θ
    @tf.function
    def profit(self, K, Z):
        return Z * tf.pow(K, self.theta)

    # investment(k,k') = k' - (1-δ)k
    @tf.function
    def investment(self, K_prime, K):
        return K_prime - (1.0 - self.delta) * K

    # psi(I,K): Adjusment cost
    @tf.function
    def investment_cost(self, I, K):
        if self.cost_mode == "None":
            return tf.zeros_like(I) * (I + K)
        elif self.cost_mode == "Cost":
            return tf.where(
                tf.not_equal(I, 0.0),
                0.5 * self.phi0 * (tf.pow(I, 2)) / K + self.phi1 * K,
                tf.zeros_like(I) * (I + K)
            )
        else:
            raise ValueError("cost_type must be 'None' or 'Cost'.")

    #  Current flow C = profit - investment cost - investment
    @tf.function
    def cashflow(self, K, Z, K_prime):
        Pi = self.profit(K, Z)
        I = self.investment(K_prime, K)
        psi = self.investment_cost(I, K)
        e = Pi- psi - I
        return e #current reward   
    
    @tf.function()
    def lifetime_reward(self, net, cur_state, T):
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")
        K_t, Z_t = cur_state
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
    def Euler_residuals(self, net,  cur_state):
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
    def Bellman_residual(self, net, cur_state, nu = 10):
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
    

    # Init the state
    def sample_state_train(self, batch_size):

        K_sample = tf.random.uniform(shape=(batch_size, 1), minval=self.K_min, maxval=self.K_max, dtype=tf.float32)
        Z_sample = tf.random.uniform(shape=(batch_size, 1), minval=self.Z_min, maxval=self.Z_max, dtype=tf.float32)
        
        #tf.print(" Initialized", batch_size, "state samples (K, Z)")
        return (K_sample, Z_sample)

    def sample_state_test(self, test_size):
      
  
        K_eval = self.K_steady * (1.0 + tf.random.normal(
        (test_size, 1), mean=0.0, stddev=0.1))
        K_eval = tf.clip_by_value(K_eval, self.K_min, self.K_max)
     
        Z_eval = tf.exp(tf.random.normal(( test_size, 1),
                                        mean=0.0,
                                        stddev=self.std_z))
        Z_eval = tf.clip_by_value(Z_eval, self.Z_min, self.Z_max)

        return (K_eval, Z_eval)
      


class RiskFree(tf.Module):
    def __init__(self,
        phi0 = 0.01,       # Convex term in the adjustment cost function
        phi1 = 0.0,            # Linear term in the adjustment cost function
        cost_type = "None",
      
        theta=0.7,         # The production function curvature (\pi(k,z) = z*k^θ)
        delta=0.08,        # Depreciation rate: fraction of capital lost each period
        r=0.04,            # Real interest rate: discounting future returns

        eta0 = 0.0,        # The fixed effects of the cost of external finance
        eta1 = 0.07,       # The linear components of the cost of external finance

        tau = 0.2,         # Tax rate

        s = 0.5,           # The fraction of the capital stock can be liquidated to repay capital
        

        rho_z=0.7,         # The persistence of z shock
        std_z=0.15,        # The standard deviation of the error term 
        ):
        super().__init__()

        # Adjustment cost parameters 
        self.phi0 = tf.constant(phi0, dtype = tf.float32)
        self.phi1 = tf.constant(phi1,dtype = tf.float32)
        self.cost_mode = cost_type

        # Economic parameters
        self.theta = tf.constant(theta, dtype=tf.float32)
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.r = tf.constant(r, dtype=tf.float32)
        self.beta = 1.0 / (1.0 + self.r)

        # Costly External Finance
        self.eta0 = tf.constant(eta0, dtype=tf.float32)
        self.eta1 = tf.constant(eta1, dtype=tf.float32)

        # Tax
        self.tau =  tf.constant(tau, dtype=tf.float32)

        # Risk
        self.s = tf.constant(s, dtype =tf.float32)
        

        # shock parameters 
        self.rho_z = tf.constant(rho_z, dtype=tf.float32)
        self.std_z = tf.constant(std_z, dtype=tf.float32)
        self.Z_min = tf.exp(-2.0 * std_z / tf.sqrt(1.0 - rho_z**2))

        # Compute steady state
        self.K_steady = (self.theta / (self.r + self.delta)) ** (1.0 / (1.0 - self.theta)) # Capital
        self.Pi_steady = self.K_steady ** self.theta # Profit
        self.I_steady = self.delta * self.K_steady # Investment
        self.E_steady = self.Pi_steady - self.I_steady # Cash flow
        self.V_steady = self.E_steady / self.r # Company Value    

        self.B_steady =  (1.0 - self.tau) * tf.pow(self.K_steady, self.theta)    

        self.Z_min = tf.exp(-2.0 * self.std_z / tf.sqrt(1.0 - self.rho_z**2))
        self.Z_max = tf.exp( 2.0 * self.std_z / tf.sqrt(1.0 - self.rho_z**2))

        self.K_min, self.K_max =  0.5 * self.K_steady, 1.5 * self.K_steady
 
        self.B_min, self.B_max = -0.5 * self.B_steady, 1.0 * self.B_steady
    
    # Model function
    # profit(k,z) = z * k^θ
    @tf.function
    def profit(self, K, Z):
        return Z * tf.pow(K, self.theta)

    # investment(k,k') = k' - (1-δ)k
    @tf.function
    def investment(self, K_prime, K):
        return K_prime - (1.0 - self.delta) * K


    # psi(I,K): Adjusment cost
    @tf.function
    def investment_cost(self, I, K):
        if self.cost_mode == "None":
            return tf.zeros_like(I) * (I + K)
        elif self.cost_mode == "Cost":
            return tf.where(
                tf.not_equal(I, 0.0),
                0.5 * self.phi0 * (tf.pow(I, 2)) / K + self.phi1 * K,
                tf.zeros_like(I) * (I + K)
            )
        else:
            raise ValueError("cost_type must be 'None' or 'Cost'.")


    # Collateral constraint
    @ tf.function
    def collateral_constraint(self, K_prime):
        RHS = (1 - self.tau) * self.Z_min * tf.pow(K_prime, self.theta) \
            + self.tau * self.delta * K_prime + self.s * K_prime
        return RHS

    
    #  Current flow C = profit - investment cost - investment
    @tf.function
    def cashflow(self,  K, K_prime, B, B_prime, Z):
        #   e = (1-tau)pi(k,z) - psi(I,K) - I + b'/(1+r(1-tau)) - b
        Pi = self.profit(K, Z)
        I = self.investment(K_prime, K)
        psi = self.investment_cost(I, K)
        e = (1 - self.tau) * Pi - psi - I + B_prime /(1.0 + self.r * (1 - self.tau)) - B 
        return e
    

    @tf.function
    def external_finance(self, e):
        #e = self.Cashflow(K, K_prime, B, B_prime, Z)
        #if isinstance(e, tuple):
            #e = e[0]
        indicator = tf.cast(e < 0.0, tf.float32)
        eta = (self.eta0 + self.eta1 * e) * indicator
        return eta

    @ tf.function
    def Reward(self, net,  K, K_prime, B, B_prime, Z):
        e = self.cashflow(K, K_prime, B, B_prime, Z)
        eta = self.external_finance(e)
        reward = e + eta
        return reward


    @tf.function()
    def lifetime_reward(self, net, cur_state, T):
        """
        Compute lifetime discounted utility for the risk-free + debt model.
        input: init state (K,, Z)
        """
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")

        K_t, B_t, Z_t = cur_state
        N = tf.shape(K_t)[0]
        reward_sum = tf.zeros((N,1), dtype=tf.float32)
    

        for t in tf.range(T):
            # Next period K‘, B‘
           
            K_next, B_next = net.policy(K_t, B_t, Z_t)
            # Current Reward
            e_t = self.cashflow(K_t, K_next, B_t, B_next, Z_t)
            eta_t = self.external_finance(e_t)
            R_t = e_t + eta_t

            reward_sum += (self.beta ** tf.cast(t, tf.float32)) * R_t

            # shock 
            eps = tf.random.normal((N, 1), mean=0.0, stddev=self.std_z)
            Z_t = tf.exp(self.rho_z * tf.math.log(Z_t) + eps)

            # State update
            K_t = K_next
            B_t = B_next
    
        return reward_sum



    @tf.function
    def Bellman_residual(self, net,  cur_state, nu_K = 10, nu_lam =10, nu_FB = 1e3):
      
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")

        K, B, Z = cur_state
        
        self.nu_K = nu_K
        self.nu_FB = nu_FB
        self.nu_lam = nu_lam

        Z_min = tf.exp(-2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))
        Z_max = tf.exp(2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))

        # current output
        K_prime, B_prime = net.policy(K, B, Z)
        Lam = net.multiplier(K, B, Z)
        V_now = net.value(K, B, Z)
    
       
        with tf.GradientTape() as tape_R:
            tape_R.watch([K_prime, B_prime])
            e = self.cashflow(K, K_prime, B, B_prime, Z)
            eta = self.external_finance(e)
            R = e + eta
        dR_dKp, dR_dBp = tape_R.gradient(R, [K_prime, B_prime])
        del tape_R

        
        eps1 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)

        def next_values(eps):
            Z_next =  tf.exp(self.rho_z * tf.math.log(Z) + eps)
            Z_next = tf.clip_by_value(Z_next, Z_min, Z_max)
            val_next = net(tf.concat([K_prime, B_prime, Z_next], axis=1))
            V_next = val_next[:, 3:4]

            with tf.GradientTape() as tapeV:
                tapeV.watch([K_prime, B_prime])
                out = net(tf.concat([K_prime, B_prime, Z_next], axis=1))[:, 3:4]
                dVnext_dKp, dVnext_dBp = tapeV.gradient(out, [K_prime, B_prime])
            return V_next, dVnext_dKp, dVnext_dBp
        
        V_next1, dVnext_dKp1, dVnext_dBp1 = next_values(eps1)
        V_next2, dVnext_dKp2, dVnext_dBp2 = next_values(eps2)

        # residuals under two shocks
        # Value function
        residual_V_eps1 = V_now - (R + self.beta * V_next1)
        residual_V_eps2 = V_now - (R + self.beta * V_next2)

        # FOC 
        residual_FOC_K_eps1 = dR_dKp + self.beta * dVnext_dKp1
        residual_FOC_K_eps2 = dR_dKp + self.beta * dVnext_dKp2

        # Constraint FOC
        residual_FOC_lam1 = self.Lam_residual(Lam, dR_dBp, dVnext_dBp1)
        residual_FOC_lam2 = self.Lam_residual(Lam, dR_dBp, dVnext_dBp2)
        
        # FB residual for constraint
        residual_FB = self.FB_residual(K_prime, B_prime, Lam)
        
        residual_V = residual_V_eps1 * residual_V_eps2
        residual_FOC_K = residual_FOC_K_eps1 * residual_FOC_K_eps2
        residual_FOC_lam = residual_FOC_lam1 * residual_FOC_lam2
        
        loss_total = tf.reduce_mean(residual_V + self.nu_K * residual_FOC_K + self.nu_lam * residual_FOC_lam + self.nu_FB * residual_FB)
        loss_V = tf.reduce_mean(tf.abs(residual_V))
        loss_FOC_K = tf.reduce_mean(tf.abs(residual_FOC_K))
        loss_FOC_lam = tf.reduce_mean(tf.abs(residual_FOC_lam))
        loss_FB = tf.reduce_mean(tf.abs(residual_FB))

        loss_dict = {
            "loss_total": loss_total,
            "loss_V": loss_V,
            "loss_FOC_K": loss_FOC_K,
            "loss_FOC_lam": loss_FOC_lam,
            "loss_FB": loss_FB,
        }


        return loss_dict

    def FB_residual(self, K_prime, B_prime, Lam):
        """ FB residual for the collateral constraint """
        g = self.collateral_constraint(K_prime) - B_prime
        FB_fun = g + Lam  - tf.sqrt(tf.pow(g, 2) + tf.pow(Lam, 2))
        FB_res = FB_fun * FB_fun
        return FB_res

    def Lam_residual(self, lam, dR_dBP, dVnext_dBP):
        """ 
        Target value for Lamda
        lamba_target = - (dR_dBP + (1 / (1 + r)) * dV_dBP)
        """
        Lam_target = - (dR_dBP + self.beta * dVnext_dBP)
        Lam_res = lam - Lam_target
        return Lam_res



    # Init the state
    def sample_state_train(self, batch_size):
        K_sample = tf.random.uniform(shape=(batch_size, 1), minval=self.K_min, maxval=self.K_max, dtype=tf.float32)

        Z_sample = tf.random.uniform(shape=(batch_size, 1), minval=self.Z_min, maxval=self.Z_max, dtype=tf.float32)

        B_sample = tf.random.uniform(shape=(batch_size, 1), minval=self.B_min, maxval=self.B_max, dtype=tf.float32,)

        
        #tf.print(" Initialized", batch_size, "state samples (K, Z)")
        return (K_sample, B_sample, Z_sample)

    def sample_state_test(self, test_size):
        # generate test data
      
        K_eval = self.K_steady * (1.0 + tf.random.normal(
        (test_size, 1), mean=0.0, stddev=0.1))
        K_eval = tf.clip_by_value(K_eval, self.K_min, self.K_max)

        B_eval = (1.0 + tf.random.normal(
        (test_size, 1), mean=0.0, stddev=0.5))
        B_eval = tf.clip_by_value(B_eval, self.B_min, self.B_max)
      
        # sample from normal distribution
        Z_eval = tf.exp(tf.random.normal(( test_size, 1),
                                        mean=0.0,
                                        stddev=self.std_z))
        Z_eval = tf.clip_by_value(Z_eval, self.Z_min, self.Z_max)


        return (K_eval, B_eval, Z_eval)

class RiskDebt(RiskFree):

    def __init__(self, prev_net,  alpha = 0., **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)
        self.alpha = alpha
        self.prev_net = prev_net  # Previous network to compute V(K', b', Z') and Policy
     
   
    @ tf.function
    def cashflow(self, K, Kp, B, Bp, Z):
        # production and investment terms
        Pi = self.profit(K, Z)                           # π(k,z)
        I = self.investment(Kp, K)                       # I = k' - (1 - delta)k
        psi = self.investment_cost(I, K)                 # psi(I,K)
        # risky rate for each (Kp, Bp, Z) state
        r_tilde = self.get_r_tilde(self.prev_net, Kp, Bp, Z)  # shape compatible with B_prime

        mask_default = self.mask_default_set(self.prev_net, Kp, Bp, Z)

        # normal
        denom_r = 1.0 + r_tilde
        debtp_normal = Bp / denom_r
        tax_normal   = (self.tau * r_tilde * Bp) / (denom_r * (1.0 + self.r))

        # default (Expected next period value <= 0)
        # No pay for old debt
        # No new debt
        # Cashflow will be the recovery value
        R_default = self.Recovery(K, Z)
        debtp_default = tf.zeros_like(Bp)
        tax_default   = tf.zeros_like(Bp) 
      
        Pi_tax= (1.0 - self.tau) * Pi
        cost = - psi - I

        e_normal  = Pi_tax + cost + tax_normal + debtp_normal - B
        e_default = R_default    # default case cashflow, recovery only
        e = e_normal * (1.0 - mask_default) + e_default * mask_default

        return e
    
    
    @ tf.function
    def Recovery(self, K, Z):
        profit = self.profit(K, Z) 
        Recovery = (1 - self.alpha) * ((1 - self.tau) * profit + (1 - self.delta) * K)
        return Recovery

    @ tf.function(reduce_retracing=True)
    def get_r_tilde(self, net, Kp, Bp, Z, N_mc=100):
        # Net: Previous network to compute V(K', b', Z')
        # Z': Next‑period shocks (Monte Carlo integration)
        
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
        R_values = self.Recovery(Kp_flat, Z_flat)

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
    def Bellman_residual(self, net,  cur_state, nu = 10):
    # Reset default states
        if not isinstance(cur_state, (tuple, list)):
            raise ValueError("Current state must be tuple or list")

        K, B, Z = cur_state
        
        self.nu = nu

        Z_min = tf.exp(-2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))
        Z_max = tf.exp(2 * self.std_z / tf.sqrt(1 - self.rho_z ** 2))

        # current output
        K_prime, B_prime = net.policy(K, B, Z)
        V_now = net.value(K, B, Z)
        
        with tf.GradientTape() as tape_R:
            tape_R.watch([K_prime, B_prime])
            e = self.cashflow(K, K_prime, B, B_prime, Z)
            eta = self.external_finance(e)
            R = e + eta
        dR_dKp, dR_dBp = tape_R.gradient(R, [K_prime, B_prime])
        del tape_R

        
        eps1 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)
        eps2 = tf.random.normal(Z.shape, mean=0.0, stddev=self.std_z)

        def next_values(eps):
            e = self.cashflow(K, K_prime, B, B_prime, Z)
            mask_default = self.mask_default_set(self.prev_net, K_prime, B_prime, Z)
            # Restart when default
            K_restart = tf.ones_like(K) * self.K_steady   # Restart 
            B_restart = tf.zeros_like(B)              # No Debt when restart

            # Combine two conditions
            Kp_eff = K_prime * (1.0 - mask_default) + K_restart * mask_default
            Bp_eff = B_prime * (1.0 - mask_default) + B_restart * mask_default

            Z_next =  tf.exp(self.rho_z * tf.math.log(Z) + eps)
            Z_next = tf.clip_by_value(Z_next, Z_min, Z_max)

            #val_next = net(tf.concat([K_prime, B_prime, Z_next], axis=1))
            V_next = net.value(Kp_eff, Bp_eff, Z_next)

            with tf.GradientTape() as tapeV:

                tapeV.watch([Kp_eff, Bp_eff])
                out = net(tf.concat([Kp_eff, Bp_eff, Z_next], axis=1))[:, 2:3]
                dVnext_dKp, dVnext_dBp = tapeV.gradient(out, [Kp_eff, Bp_eff])

            return V_next, dVnext_dKp, dVnext_dBp
        
        V_next1, dVnext_dKp1, dVnext_dBp1 = next_values(eps1)
        V_next2, dVnext_dKp2, dVnext_dBp2 = next_values(eps2)

        # residuals under two shocks
        # Value function
        residual_V_eps1 = V_now - (R + self.beta * V_next1)
        residual_V_eps2 = V_now - (R + self.beta * V_next2)

        # FOC 
        residual_FOC_K_eps1 = dR_dKp + self.beta * dVnext_dKp1
        residual_FOC_K_eps2 = dR_dKp + self.beta * dVnext_dKp2

        residual_FOC_B_eps1 = dR_dBp + self.beta * dVnext_dBp1
        residual_FOC_B_eps2 = dR_dBp + self.beta *  dVnext_dBp2

        residual_V = residual_V_eps1 * residual_V_eps2
        residual_FOC_K = residual_FOC_K_eps1 * residual_FOC_K_eps2
        residual_FOC_B = residual_FOC_B_eps1 * residual_FOC_B_eps2
        
        loss_total = tf.reduce_mean(residual_V + self.nu * residual_FOC_K + self.nu * residual_FOC_B)
        loss_V = tf.reduce_mean(tf.abs(residual_V))
        loss_FOC_K = tf.reduce_mean(tf.abs(residual_FOC_K))
        loss_FOC_B = tf.reduce_mean(tf.abs(residual_FOC_B))

        loss_dict = {
            "loss_total": loss_total,
            "loss_V": loss_V,
            "loss_FOC_K": loss_FOC_K,
            "loss_FOC_B": loss_FOC_B,
        }

        return loss_dict

