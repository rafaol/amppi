# TODO:
# - Vectorize code

import math
import torch
import pyro
import pyro.distributions as dist
import gym
from torch.distributions import constraints
# Local package
import amppi

ENV_NAME = "CartPole-v1"
TIMESTEPS = 20  # T
N_SAMPLES = 200  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
LAMBDA_ = 1

class CartPoleModel:
    """Refer to A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike
    adaptive elements that can solve difficult learning control problems,”
    IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13,
    pp. 834–846, Sept./Oct. 1983.
    """
    def __init__(self, g=9.8, m_c_loc=1.0, m_c_scale=0.1, m_p=0.1, l_loc=0.5,
                 l_scale=0.1, mu_c=0.0005, f_mag=10.0, mu_p=0.000002, dt=0.02):
        self.g = g #m/sˆ2
        self.m_p = m_p #kg
        self.m_c_loc = pyro.param("m_c_loc", torch.tensor(m_c_loc),
                                  constraint=constraints.positive)
        self.m_c_scale = pyro.param("m_c_scale", torch.tensor(m_c_scale),
                                    constraint=constraints.positive)
        self.l_loc = pyro.param("l_loc", torch.tensor(l_loc),
                                constraint=constraints.positive)
        self.l_scale = pyro.param("l_scale", torch.tensor(l_scale),
                                  constraint=constraints.positive)
        self.mu_c = mu_c
        self.mu_p = mu_p
        self.f_mag = f_mag # u = f_mag N applied to the cart's center of mass
        self.dt = dt #s

    def sample_params(self):
        m = pyro.sample("m_sample", dist.Normal(self.m_c_loc, self.m_c_scale))
        l = pyro.sample("l_sample", dist.Normal(self.l_loc, self.l_scale))
        return torch.tensor([m, l])

    def step(self, state, action, params):
        x, x_d, theta, theta_d = state
        m_c, l = params
        u = action*self.f_mag

        mass = m_c+self.m_p #total mass
        pm = self.m_p*l # polemass
        cart_friction = self.mu_c*torch.sign(x_d)
        pole_friction = (self.mu_p*theta_d)/pm
        factor = (u + pm*torch.sin(theta)*theta_d**2 - cart_friction)/mass
        th_num = (self.g*torch.sin(theta) - torch.cos(theta)*factor
                  - pole_friction)
        th_den = l*(4.0/3-(self.m_p*torch.cos(theta)**2)/mass)
        theta_dd = th_num/th_den

        x_dd = factor - pm*theta_dd*torch.cos(theta)/mass
        delta = torch.tensor([x_d, x_dd, theta_d, theta_dd])*self.dt
        return state+delta

    def compute_terminal_cost(self, state):
        print("terminal_cost")
        terminal = state[0] < -2.4 \
                   or state[0] > 2.4 \
                   or state[2] < -12*2*math.pi/360 \
                   or state[2] > 12*2*math.pi/360
        return 1000000 if terminal else 0

    def compute_state_cost(self, state):
        # Original cost function on MPPI paper
        print("state_cost")
        return (state[0]**2 + 500*torch.sin(state[2])**2
                + 1*state[1]**2 + 1*state[3]**2)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state = env.reset()
    state = torch.tensor(state)
    model = CartPoleModel(mu_p=0, mu_c=0)
    controller = amppi.AMPPI(obs_space=env.observation_space,
                             act_space=env.action_space,
                             K=N_SAMPLES,
                             T=TIMESTEPS,
                             lambda_=LAMBDA_)
    step = 0
    while True:
        env.render()
        u, tau, cost, omega = controller.act(model, state)
        action = 1 if u>0 else 0  # converts to a binary scalar
        if not step%10:
            print("Step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Current state: x={0[0]}, theta={0[2]}".format(state))
            print("Next actions 4 actions: {}".format(
                  controller.U[:4].flatten()))
        state, _, done, _ = env.step(action)
        state = torch.tensor(state)
        if done:
            print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Last state: x={0[0]}, theta={0[2]}".format(state))
            print("Next actions: {}".format(controller.U.flatten()))
            env.close()
            break
        step += 1
