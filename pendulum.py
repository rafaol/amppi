# TODO: Vectorize code

import math
import torch
import pyro
import pyro.distributions as dist
import gym
from torch.distributions import constraints
# local package
import amppi

ENV_NAME = "Pendulum-v0"
TIMESTEPS = 20   # T
N_SAMPLES = 200  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
LAMBDA_ = 1

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

    def sample_params(self):
        m = pyro.sample("m_sample", dist.Normal(self.m_loc, self.m_scale))
        l = pyro.sample("l_sample", dist.Normal(self.l_loc, self.l_scale))
        return torch.tensor([m, l])

    def step(self, state, action, params):
        theta, theta_d = state
        m, l = params
        g = self.g
        dt = self.dt

        theta_d = theta_d + dt*(-3*g/(2*l) * torch.sin(theta+math.pi)
                                + 3./(m*l**2)*action)
        theta = theta + theta_d*dt  # Use new theta_d
        theta_d = torch.clamp(theta_d, -self.max_speed, self.max_speed)
        return torch.tensor([theta, theta_d])

    def compute_terminal_cost(self, state):
        return 0

    def compute_state_cost(self, state):
        # Note that theta may range beyond 2*pi
        theta, theta_d = state
        return (10*(torch.cos(theta)-1)**2 + .01*theta_d**2).reshape(-1, 1)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    env.reset()
    state = env.state
    state = torch.tensor(state)
    model = PendulumModel()
    controller = amppi.AMPPI(obs_space=env.observation_space,
                             act_space=env.action_space,
                             K=N_SAMPLES,
                             T=TIMESTEPS,
                             lambda_=LAMBDA_,
                             eps_scale=torch.eye(1)*10)
    step = 0
    while step<200:
        env.render()
        action, tau, cost, omega = controller.act(model, state)
        if not step%25:
            print("Step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Current state: "
                  "theta={0[0]}, theta_dot={0[1]}".format(state))
            print("Next actions 4 actions: {}" \
                  .format(controller.U[:4].flatten()))
        # input("Press Enter to continue...")
        _, _, done, _ = env.step(action)
        state = env.state
        state = torch.tensor(state)
        if done:
            print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Last state: "
                  "theta={0[0]}, theta_dot={0[1]}".format(state))
            print("Next actions: {}".format(controller.U.flatten()))
            env.close()
            break
        step += 1
