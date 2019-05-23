# TODO: Understand why results are so sensitive on the cost function

import math
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
    def __init__(self, g=9.8, m_loc=1.0, m_scale=0.1, l_loc=1.0, l_scale=0.1,
                 max_torque=2., max_speed=8, dt=0.05):
        self.g = g  #m/s^2
        self.m_loc = pyro.param("m_loc", torch.tensor(m_loc),
                                constraint=constraints.positive)  # kg
        self.m_scale = pyro.param("m_scale", torch.tensor(m_scale),
                                  constraint=constraints.positive)  # kg
        self.l_loc = pyro.param("l_loc", torch.tensor(m_loc),
                                constraint=constraints.positive)  # m
        self.l_scale = pyro.param("l_scale", torch.tensor(m_scale),
                                  constraint=constraints.positive)  # m
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
        # Use detach() to set require_grad=False
        m = pyro.sample("m_sample", dist.Normal(self.m_loc, self.m_scale),
                        sample_shape=sample_shape).view(-1, 1).detach()
        l = pyro.sample("l_sample", dist.Normal(self.l_loc, self.l_scale),
                        sample_shape=sample_shape).view(-1, 1).detach()
        return torch.cat((m, l), dim=1)

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


def run_amppi(steps=200, verbose=True, msg=10):
    def terminal_cost(states):
        return torch.zeros(states.shape[0])

    def state_cost(states):
        # Note that theta may range beyond 2*pi
        theta, theta_d = states.chunk(2, dim=1)
        return (10*(theta.cos()-1)**2 + 0.1*theta_d**2)

    states = []
    actions = []
    costs = []
    env = gym.make(ENV_NAME)
    env.reset()
    state = torch.tensor(env.state)
    model = PendulumModel(g=10)
    controller = amppi.AMPPI(obs_space=env.observation_space,
                             act_space=env.action_space,
                             K=N_SAMPLES,
                             T=TIMESTEPS,
                             lambda_=LAMBDA_,
                             cov=torch.eye(1)*EPS_SCALE,
                             term_cost_fn=terminal_cost,
                             inst_cost_fn=state_cost,
                             sampling='extended'
                             ctrl_cost=CONTROL_COST)
    controller.n = 2  # Not using default Gym obs_space dimension
    step = 0
    while step<steps:
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
        states.append(state.tolist())
        actions.append(action.item())
        costs.append(cost.item())
        step += 1

    print("Last step {0}: action taken {1:.2f}, cost {2:.2f}"\
          .format(step, float(action), cost))
    print("Last state: theta={0[0]}, theta_dot={0[1]}".format(state))
    print("Next actions: {}".format(controller.U.flatten()))
    env.close()
    return states, actions, costs


if __name__ == "__main__":
    s, a, c = run_amppi(steps=300, verbose=False)
