import torch

class AMPPI:
    """ Implements an MMPI controller as defined in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning', for
    OpenAI Gym environments.
    :param env: an OpenAi Gym environment
    :param obs_space: an OpenAI bounding box for the state space
    :param act_space: an OpenAI bounding box for the action space, the min_u
    (max_u) minimum (maximum) control value is derived from act_space
    :param K: number of sampled trajectories
    :param T: number of timesteps in the control horizon
    :param U: control matrix of the form [u_0, ..., u_T]
    :param lambda_: controller regularization parameter (default: 1.0)
    :param u_init: initial control action
    :param eps_scale: covariance matrix of the Gaussian noise (default: None)

    Note:
    The noise epsilon will be clipped according to min_u and max_u, effectively
    epsilon <= max_u-min_u
    """
    # params = torch.tensor([1., 1.])
    def __init__(self, obs_space, act_space, K, T, lambda_=1.0,
                 eps_scale=None):
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
            # torch.clamp assumes same boundaries for all tensor dims
            self.max_u = float(act_space.high)
            self.min_u = float(act_space.low)
        self.n = obs_space.shape[0] if obs_space.shape else 1
        if eps_scale is None:
            eps_scale = torch.eye(self.m)
        eps_loc = torch.empty(self.m)
        self.eps_dist = torch.distributions.multivariate_normal \
                             .MultivariateNormal(eps_loc, eps_scale)
        self.eps_pre = torch.inverse(eps_scale)
        self.U = torch.empty((self.T, self.m))

    def get_trajectory_cost(self, model, state, tau):
        # TODO: figure out why a floating point exception:8 is happening
        """ Computes the total cost of a single control trajectory
        Keyword arguments:
        state -- the trajectory initial state
        tau -- a sampled trajectory
        """
        S = 0
        params = model.sample_params()
        for t in range(self.T):
            # Takes a step in the dynamical model and sets new state
            state = model.step(state, tau[t, :], params)
            S += model.compute_state_cost(state)
        # For 1-d control we can vectorize the cross-term
        # TODO: Figure out why cross-term is adding a bias to the controller
        # S += self.lambda_*self.eps_pre*self.U.T@(tau-self.U)  # clipped epsilon
        S += model.compute_terminal_cost(state)
        return S

    def sample_trajectories(self, model, state):
        # eps shape is K x T x m
        eps = self.eps_dist.sample(sample_shape=[self.K, self.T])
        tau = torch.add(eps, self.U)
        tau = torch.clamp(tau, self.min_u, self.max_u)
        # if self.discrete:
        #     one = torch.tensor(1)
        #     tau = torch.where(tau>0, one, -one)  # 1 if >0, -1 otherwise
        cost = torch.empty(self.K)
        eta = 0
        for k in range(self.K):
            cost[k] = self.get_trajectory_cost(model, state, tau[k])
        beta = torch.min(cost)
        eta = torch.sum(torch.exp(-1/self.lambda_*(cost-beta)))  # scalar
        omega = torch.exp(-1/self.lambda_*(cost-beta))/eta  # tensor of size k
        return tau, omega, beta

    def act(self, model, state):
        tau, omega, beta = self.sample_trajectories(model, state)
        eps = torch.add(tau, -self.U)  # clipped epsilon
        # use torch.tensordot to multiply omega and epsilon for all T
        self.U = torch.tensordot(omega, eps, dims=1)
        # Even though tau has been clipped, epsilon can be of mag (u_max-u_min)
        # so we need to clip U again
        self.U = torch.clamp(self.U, self.min_u, self.max_u)
        action = self.U[0, :]
        self.U = torch.roll(self.U, -1, 0)  # Shifts U_t+1 to U_t
        self.U[self.T-1, :] = torch.empty_like(self.U[self.T-1, :])
        return action, tau, beta, omega
