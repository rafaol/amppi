# TODO: Understand why results are so sensitive on the cost function

import math
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
import gym
from torch.distributions import constraints
import amppi

ENV_NAME = "Pendulum-v0"
TIMESTEPS = 20   # T
N_SAMPLES = 500  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
LAMBDA_ = 1
CONTROL_COST = False
EPS_SCALE = 5
LOC1 = 0.7
LOC2 = 1.3
SCALE = 0.1

class PendulumModel:
    """For more information refer to OpenAI Gym environment at:
    https://gym.openai.com/envs/Pendulum-v0/
    :param g: gravity in m/s^2 (default: 9.8)
    :param m_loc: mean of the pendulum mass in kg (default: 1.0)
    :param m_scale: variance of the pendulum mass (default: 0.25)
    :param l_loc: mean of the pendulum length in m (default: 1.0)
    :param l_scale: variance of the pendulum length (default: 0.25)
    :param max_torque: maximum control torque in N/m (default: 2.0)
    :param max_speed: maximum rotational speed in m/s (default: 8)
    :param d_t: timestep length for each update in s (default: 0.05)
    """
    def __init__(self, params_dist, g=9.8, max_torque=2., max_speed=8,
                 dt=0.05):
        self.g = g  #m/s^2
        self.params_dist = params_dist
        self.max_torque = max_torque  # N/m
        self.max_speed = max_speed  # m/s
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
        :param params: a tensor with the sampled model parameters for these
        trajectories of shape [param1, param2, ..., paramN] and length 1 or K
        """
        # Assigning states and params, keeping their dims
        theta, theta_d = states.chunk(2, dim=1)
        m, l = params.chunk(2, dim=1)
        g = self.g
        dt = self.dt

        theta_d = theta_d + dt*(-3*g/(2*l)*(theta+math.pi).sin()
                                + 3./(m*l**2)*actions)
        theta = theta + theta_d*dt  # Use new theta_d
        theta_d = torch.clamp(theta_d, -self.max_speed, self.max_speed)
        return torch.cat((theta, theta_d), dim=1)


def run_amppi(model, init_state, steps=200, verbose=True, steps_per_msg=20,
              render=False):
    """Runs the simulation loop."""
    def terminal_cost(states):
        return torch.zeros(states.shape[0])

    def state_cost(states):
        # Note that theta may range beyond 2*pi
        theta, theta_d = states.chunk(2, dim=1)
        return 10*(theta.cos()-1)**2 + 0.2*theta_d**2

    states = torch.tensor([])
    actions = torch.tensor([])
    costs = torch.tensor([])
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
    controller.n = 2  # Not using default Gym obs_space dimension
    step = 0
    while step<steps:
        if render:
            env.render()
        action, cost = controller.act(model, state)
        if verbose and not step%msg:
            print("Step {0}: action taken {1:.2f}, cost {2:.2f}"\
                  .format(step, float(action), cost))
            print("Current state: "
                  "theta={0[0]}, theta_dot={0[1]}".format(state))
            print("Next actions 4 actions: {}" \
                  .format(controller.U[:4].flatten()))
        _, _, done, _ = env.step(action)
        state = torch.tensor(env.state)
        if done:
            env.close()
            break
        states = torch.cat((states, state.reshape(1, -1)))
        actions = torch.cat((actions, action.reshape(1, -1)))
        costs = torch.cat((costs, cost.reshape(-1)))
        step += 1
    if verbose:
        print("Last step {0}: action taken {1:.2f}, cost {2:.2f}"\
              .format(step, float(action), cost))
        print("Last state: theta={0[0]}, theta_dot={0[1]}".format(state))
        print("Next actions: {}".format(controller.U.flatten()))
    env.close()
    return states, actions, costs


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

    acc_costs = torch.tensor([])
    for i in range(10):
        init_state = torch.empty(2).uniform_(-math.pi, math.pi)
        cost_sum = torch.tensor([])
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
            model = PendulumModel(g=10, params_dist=params_model)
            s, a, c = run_amppi(model, init_state, steps=300,
                                verbose=False, render=False)
            cost_sum = torch.cat((cost_sum, c.sum().reshape(-1)))
        acc_costs = torch.cat((acc_costs, cost_sum.reshape(1, -1)), dim=0)
    # Plot results
    width = 0.3
    ind0 = torch.arange(0, 20., 2.)
    ind1 = ind0 + width
    ind2 = ind1 + width
    ind3 = ind2 + width
    plt.bar(ind0, acc_costs[:, 0], width, label='Perfect model')
    plt.bar(ind1, acc_costs[:, 1], width, label='Sigma {:0.1f}'.format(0.1))
    plt.bar(ind2, acc_costs[:, 2], width, label='Sigma {:0.1f}'.format(0.2))
    plt.bar(ind3, acc_costs[:, 3], width, label='Sigma {:0.1f}'.format(0.3))
    plt.ylabel('Accumulated cost')
    x_labels = ['Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4',
                'Seed 5', 'Seed 6', 'Seed 7', 'Seed 8', 'Seed 9']
    plt.xticks((ind1+ind2)/2, x_labels)
    plt.legend()
    plt.show()
