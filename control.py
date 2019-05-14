import numpy as np

class AMPPI:
    """ Implements an MMPI controller as defined in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning', for
    OpenAI Gym environments.

    Keyword arguments:
    env -- an OpenAi Gym environment
    obs_space -- an OpenAI bounding box for the state space
    act_space -- an OpenAI bounding box for the action space, the min_u (max_u)
    minimum (maximum) control value is derived from act_space
    K -- number of sampled trajectories
    T -- number of timesteps in the control horizon
    U -- control matrix of the form [u_0, ..., u_T]
    lambda_ -- controller temperature
    u_init -- initial control action
    eps_loc -- mean vector of the Gaussian noise
    eps_scale -- covariance matrix of the Gaussian noise

    Note:
    The noise epsilon will be clipped according to min_u and max_u, effectively
    epsilon <= max_u-min_u
    """

    def __init__(self, obs_space, act_space, K, T, lambda_=1.0,
                 eps_loc=[0], eps_scale=[1]):
        self.K = K
        self.T = T
        self.lambda_ = lambda_
        self.discrete = False if act_space.shape else True
        if self.discrete:
            self.m = 1
            self.max_u = 1
            self.min_u = -1  # to make control costs symmetric when binary
        else:
            self.m = act_space.shape[0]
            self.max_u = act_space.high
            self.min_u = act_space.low
        self.n = obs_space.shape[0] if obs_space.shape else 1
        self.eps_loc = np.array(eps_loc)
        self.eps_cov = np.eye(self.m)*eps_scale
        self.eps_pre = np.linalg.inv(self.eps_cov)
        self.U = np.zeros((self.T, self.m))


    def get_trajectory_cost(self, model, state, tau):
        """ Computes the total cost of a single control trajectory
        Keyword arguments:
        state -- the trajectory initial state
        tau -- a sampled trajectory
        """
        S = 0
        for t in range(self.T):
            # Takes a step in the dynamical model and sets new state
            state = model.step(state, tau[t, :])
            S += model.compute_state_cost(state)
        # For 1-d control we can vectorize the cross-term
        # TODO: Figure out why cross-term is adding a bias to the controller
        # S += self.lambda_*self.eps_pre*self.U.T@(tau-self.U)  # clipped epsilon
        S += model.compute_terminal_cost(state)
        return S


    def sample_trajectories(self, model, state):
        # epsilon shape is K x T x m
        epsilon = np.random.multivariate_normal(
            mean=self.eps_loc,
            cov=self.eps_cov,
            size=(self.K, self.T)
        )
        tau = np.add(epsilon, self.U)
        tau = np.clip(tau, self.min_u, self.max_u)
        if self.discrete:
            tau = np.where(tau>0, 1, -1)  # 1 if >0, -1 otherwise
        cost = np.zeros(self.K)
        eta = 0
        for k in range(self.K):
            cost[k] = self.get_trajectory_cost(model, state, tau[k])
        beta = np.min(cost)
        eta = np.sum(np.exp(-1/self.lambda_*(cost-beta)))  # returns a scalar
        omega = np.exp(-1/self.lambda_*(cost-beta))/eta  # vector of length k
        return tau, omega, beta


    def act(self, model, state):
        tau, omega, beta = self.sample_trajectories(model, state)
        epsilon = np.add(tau, -self.U)  # clipped epsilon
        # use np.tensordot to multiply omega and epsilon for all T
        self.U = np.tensordot(omega, epsilon, axes=([0], [0]))
        # Even though tau has been clipped, epsilon can be of mag (u_max-u_min)
        # so we need to clip U again
        self.U = np.clip(self.U, self.min_u, self.max_u)
        action = self.U[0, :]
        self.U = np.roll(self.U, -1, axis=0)  # Shifts U_t+1 to U_t
        self.U[self.T-1, :] = np.zeros_like(self.U[self.T-1, :])
        return action, tau, beta, omega
