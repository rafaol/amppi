# TODO: Experiment on the control cost as a regularization factor
# TODO: Can multiple control update steps reduce noise?
# TODO: Does cost normalisation improve trajectory averaging

import torch

class AMPPI:
    """ Implements an MMPI controller as defined in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning', for
    OpenAI Gym environments.
    :param obs_space: an OpenAI bounding box for the state space
    :param act_space: an OpenAI bounding box for the action space, the min_u
    (max_u) minimum (maximum) control value is derived from act_space
    :param K: number of sampled trajectories
    :param T: number of timesteps in the control horizon
    :param lambda_: controller regularization parameter (default: 1.0)
    :param cov: covariance matrix of the Gaussian noise (default: None)
    :param sampling: can be set to either 'single' or 'extended', controls if
    the forward model parameters samples are common or not to all trajectories
    (default: 'single')
    :param ctrl_cost: bool, if True the control actions cost is considered,
    otherwise they are disregarded (default: False)

    Note:
    The noise epsilon will be clipped according to min_u and max_u, effectively
    epsilon <= max_u-min_u
    """
    def __init__(self, obs_space, act_space, K, T, lambda_=1.0, cov=None,
                 sampling='single', inst_cost_fn=None, term_cost_fn=None,
                 ctrl_cost=False):
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
        if cov is None:
            cov = torch.eye(self.m)
        eps_loc = torch.zeros(self.m)
        self.eps_dist = torch.distributions.multivariate_normal \
                             .MultivariateNormal(eps_loc, cov)
        self.pre = torch.inverse(cov)
        self.U = torch.zeros((self.T, self.m))
        # Consider the cross term control cost?
        self.ctrl = 1 if ctrl_cost else 0
        self.sample_shape = [1] if sampling=='single' else [K]

        # At least one cost function
        if inst_cost_fn is None and term_cost_fn is None:
            raise ValueError("Specify at least one cost function")

        def no_cost_fn(ss):
            return torch.zeros(ss.m, 1)

        if inst_cost_fn is None:
            self._inst_cost_fn = no_cost_fn
        else:
            self._inst_cost_fn = inst_cost_fn
        if term_cost_fn is None:
            self._term_cost_fn = no_cost_fn
        else:
            self._term_cost_fn = term_cost_fn

    def _roll(self, step):
        self.U = torch.roll(self.U, -step, 0)  # Shifts U_t+1 to U_t
        self.U[-step:, :] = torch.zeros_like(self.U[-step:, :])

    def _discretize(self, t):
            one = torch.tensor(1.0)
            return torch.where(t>0, one, -one)

    def _sample_trajectories(self, model, state):
        # Sample trajectories, eps shape is K x T x m
        eps = self.eps_dist.sample(sample_shape=[self.K, self.T])
        actions = torch.add(eps, self.U)
        actions = torch.clamp(actions, self.min_u, self.max_u)
        if self.discrete:
            actions = self._discretize(actions)
        eps = torch.add(actions, -self.U)  # clipped epsilon
        states = torch.zeros(self.K, self.T+1, self.n)
        states[:, 0, :] = state.repeat(self.K, 1)
        params = model.sample_params(self.sample_shape)
        for t in range(self.T):
            states[:, t+1, :] = model.step(states[:, t, :], actions[:, t, :],
                                           params)
        return actions, states, eps

    def act(self, model, state):
        """Computes the next control action and the incurred cost. Updates the
        controller next control actions U.
        :param model: a forward model with a step(states, actions, params)
        function to compute the next state for a set of trajectories
        :param state: a tensor with the system initial state
        """
        actions, states, eps = self._sample_trajectories(model, state)
        # Estimate trajectories cost
        # Need to use reshape instead of view because slice is not contiguous
        inst_costs = self._inst_cost_fn(states[:, 1:, :].reshape(-1, self.n))\
                     .view(self.K, self.T).sum(dim=1)
        term_costs = self._term_cost_fn(states[:, -1, :]).view(self.K)
        # To compute ctrl_costs in a single batch for all T and all K, first we
        # compute the T x m cost matrix, result size is K x T x T. Then take 
        # the trace of the T x T matrices, result is dim K.
        ctrl_costs = self.lambda_*((self.U@self.pre)@eps.transpose(1, 2))\
                     .diagonal(dim1=1, dim2=2).sum(dim=1)*self.ctrl
        costs = term_costs + inst_costs + ctrl_costs  # shape is K
        beta = torch.min(costs)
        eta = torch.exp(-1/self.lambda_*(costs-beta)).sum()  # scalar
        omega = torch.exp(-1/self.lambda_*(costs-beta))/eta  # tensor of size K
        # use torch.tensordot to multiply omega and epsilon for all T
        self.U += torch.tensordot(omega, eps, dims=1)
        # Even though tau has been clipped, epsilon can be of mag (u_max-u_min)
        # so we need to clip U again
        self.U = torch.clamp(self.U, self.min_u, self.max_u)
        action = self.U[0, :]
        # self._roll(step=1)
        cost = omega.dot(costs)
        return action, cost, omega
