import math
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import gym
import torch
from torch.distributions import constraints
import amppi

ENV_NAME = "CartPole-v1"
TIMESTEPS = 20  # T
N_SAMPLES = 1000  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
LAMBDA_ = 1
CONTROL_COST = False
EPS_SCALE = 2
LOC1 = 0.7
LOC2 = 1.3
SCALE = 0.1

class CartPoleModel:
    """Refer to A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike
    adaptive elements that can solve difficult learning control problems,”
    IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13,
    pp. 834–846, Sept./Oct. 1983.
    """
    def __init__(self, params_dist, g=9.8, f_mag=10.0, m_p=0.1,
                 mu_c=0.0005, mu_p=0.000002, dt=0.02):
        self.g = g #m/sˆ2
        self.params_dist = params_dist
        self.m_p = m_p #kg
        self.mu_c = mu_c
        self.mu_p = mu_p
        self.f_mag = f_mag # u = f_mag N applied to the cart's center of mass
        self.dt = dt #s

    def sample_params(self, sample_shape=[1]):
        """Samples parameters for the forward model.
        :param sample_shape: a list with the length of the parameter vector.
        Must be either [1] for a single set of parameters per time step for all
        trajectories, or [K] for individual set o parameters for each
        trajectory at each time step (default: [1]).
        """
        return self.params_dist(sample_shape)

    def step(self, states, actions, params):
        """Returns the next state for a set of trajectories.
        :param states: a tensor of the inital state stacked into a shape K x n
        :param actions: a tensor of control actions of shape K x m
        :param params: the sampled model parameters for these trajectories
        """
        x, x_d, theta, theta_d = states.chunk(4, dim=1)
        m_c, l = params.chunk(2, dim=1)
        u = actions*self.f_mag

        mass = m_c+self.m_p #total mass
        pm = self.m_p*l # polemass
        cart_friction = self.mu_c*x_d.sign()
        pole_friction = (self.mu_p*theta_d)/pm
        factor = (u + pm*theta.sin()*theta_d**2 - cart_friction)/mass
        tdd_num = (self.g*theta.sin() - theta.cos()*factor -  pole_friction)
        tdd_den = l*(4.0/3-(self.m_p*theta.cos()**2)/mass)
        theta_dd = tdd_num/tdd_den

        x_dd = factor - pm*theta_dd*torch.cos(theta)/mass
        delta = torch.cat((x_d, x_dd, theta_d, theta_dd), dim=1)*self.dt
        return states+delta


def run_amppi(model, init_state, steps=200, verbose=True, steps_per_msg=20,
              render=False):
    """Runs the simulation loop."""
    def terminal_cost(states):
        terminal = (states[:, 0] < -2.4) \
                   + (states[:, 0] > 2.4) \
                   + (states[:, 2] < -12*2*math.pi/360) \
                   + (states[:, 2] > 12*2*math.pi/360).clamp(0, 1)
        return (terminal*1000000).type(torch.float)

    def state_cost(states):
        # Original cost function on MPPI paper
        return (states[:, 0]**2 + 500*torch.sin(states[:, 2])**2
                + 1*states[:, 1]**2 + 1*states[:, 3]**2)

    states = torch.tensor([])
    actions = torch.tensor([])
    acc_reward = 0
    env = gym.make(ENV_NAME)
    env.reset()
    env.env.state = init_state
    state = init_state
    controller = amppi.AMPPI(obs_space=env.observation_space,
                             act_space=env.action_space,
                             K=N_SAMPLES,
                             T=TIMESTEPS,
                             lambda_=LAMBDA_,
                             cov=torch.eye(1)*EPS_SCALE,
                             term_cost_fn=terminal_cost,
                             inst_cost_fn=state_cost,
                             sampling='extended',
                             ctrl_cost=CONTROL_COST)
    step = 0
    while step<steps:
        if render:
            env.render()
        action, cost = controller.act(model, state)
        actions = torch.cat((actions, action.reshape(1, -1)))
        action = 1 if action>0 else 0  # converts to a binary scalar for Gym
        if verbose and not step%msg:
            print("Step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Current state: x={0[0]}, theta={0[2]}".format(state))
            print("Next actions 4 actions: {}".format(
                  controller.U[:4].flatten()))
        state, reward, done, _ = env.step(action)
        state = torch.tensor(state)
        acc_reward += reward
        if done:
            break
        states = torch.cat((states, state.reshape(1, -1)))
        step += 1
    if verbose:
        print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
        print("Last state: x={0[0]}, theta={0[2]}".format(state))
        print("Next actions: {}".format(controller.U.flatten()))
    env.close()
    return states, actions, acc_reward


if __name__ == "__main__":
    def params_model(sample_shape):
        """Function to sample the model paramaters."""
        # Use detach() to set require_grad=False
        loc1 = pyro.param("loc1", torch.tensor(LOC1),
                          constraint=constraints.positive)
        loc2 = pyro.param("loc2", torch.tensor(LOC2),
                          constraint=constraints.positive)
        scale = pyro.param("scale", torch.tensor(SCALE),
                           constraint=constraints.positive)
        dist1 = dist.Normal(loc1, scale)
        dist2 = dist.Normal(loc2, scale)
        m = (dist1.sample(sample_shape) + dist2.sample(sample_shape))/2
        m = m.view(-1, 1).detach()
        l = (dist1.sample(sample_shape) + dist2.sample(sample_shape))/2
        l = l.view(-1, 1).detach()
        return torch.cat((m, l), dim=1)

    acc_rewards = torch.tensor([])
    for i in range(10):
        init_state = torch.empty(4).uniform_(-0.05, 0.05)
        seed_rwd = torch.tensor([])
        for j in range(4):
            if j == 0:
                LOC1 = 1.
                LOC2 = 1.
            else:
                LOC1 = 0.7
                LOC2 = 1.3
            SCALE = float(j/10)
            print("Seed {0}: Params dist(Loc1={1}, Loc2={2}, "
                  "Sigma={3:0.1f})".format(i, LOC1, LOC2, SCALE))
            model = CartPoleModel(mu_p=0, mu_c=0, params_dist=params_model)
            s, a, r = run_amppi(model, init_state, steps=300, verbose=False,
                                render=False)
            seed_rwd = torch.cat((seed_rwd, torch.tensor(r).reshape(-1)))
        acc_rewards =  torch.cat((acc_rewards, seed_rwd.reshape(1, -1)), dim=0)
    # Plot results
    width = 0.3
    ind0 = torch.arange(0, 20., 2.)
    ind1 = ind0 + width
    ind2 = ind1 + width
    ind3 = ind2 + width
    plt.bar(ind0, acc_rewards[:, 0], width, label='Perfect model')
    plt.bar(ind1, acc_rewards[:, 1], width, label='Sigma {:0.1f}'.format(0.1))
    plt.bar(ind2, acc_rewards[:, 2], width, label='Sigma {:0.1f}'.format(0.2))
    plt.bar(ind3, acc_rewards[:, 3], width, label='Sigma {:0.1f}'.format(0.3))
    plt.ylabel('Accumulated reward')
    x_labels = ['Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4',
                'Seed 5', 'Seed 6', 'Seed 7', 'Seed 8', 'Seed 9']
    plt.xticks((ind1+ind2)/2, x_labels)
    plt.legend()
    plt.show()
