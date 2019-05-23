import math
import torch
import pyro
import pyro.distributions as dist
import gym
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

    def sample_params(self, sample_shape=[1]):
        m = pyro.sample("m_sample", dist.Normal(self.m_c_loc, self.m_c_scale),
                        sample_shape=sample_shape).view(-1, 1).detach()
        l = pyro.sample("l_sample", dist.Normal(self.l_loc, self.l_scale),
                        sample_shape=sample_shape).view(-1, 1).detach()
        return torch.cat((m, l), dim=1)

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


def run_amppi(steps=200, verbose=True, msg=10):
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

    env = gym.make(ENV_NAME)
    env.reset()
    state = torch.tensor(env.state)
    model = CartPoleModel(mu_p=0, mu_c=0)
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
    step = 0
    while step<steps:
        env.render()
        action, cost = controller.act(model, state)
        action = 1 if action>0 else 0  # converts to a binary scalar for Gym
        if verbose and not step%msg:
            print("Step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Current state: x={0[0]}, theta={0[2]}".format(state))
            print("Next actions 4 actions: {}".format(
                  controller.U[:4].flatten()))
        state, _, done, _ = env.step(action)
        state = torch.tensor(state)
        if done:
            break
        step += 1
    print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
    print("Last state: x={0[0]}, theta={0[2]}".format(state))
    print("Next actions: {}".format(controller.U.flatten()))
    env.close()


if __name__ == "__main__":
    run_amppi(steps=500, msg=20)
