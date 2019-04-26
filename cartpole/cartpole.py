# TODO:
# - Vectorize code

import gym
import numpy as np
import torch
import pyro

ENV_NAME = "CartPole-v1"
TIMESTEPS = 10  # T
N_SAMPLES = 100  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

EPSILON_LOC = 0
EPSILON_SCALE = 10
LAMBDA_ = 1

class CartPoleModel:
    """Refer to A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike
    adaptive elements that can solve difficult learning control problems,”
    IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13,
    pp. 834–846, Sept./Oct. 1983.
    """
    def __init__(self, g=9.8, m_c=1.0, m_p=0.1, l=0.5, mu_c=0.0005,
                 f_mag=10.0, mu_p=0.000002, dt=0.02):
        self.g = g #m/sˆ2
        self.m_c = m_c #kg
        self.m_p = m_p #kg
        self.l = l #m
        self.mu_c = mu_c
        self.mu_p = mu_p
        self.f_mag = f_mag # u = f_mag N applied to the cart's center of mass
        self.dt = dt #s

    def step(self, state, action):
        x, x_d, theta, theta_d = state
        # u = self.f_mag if action==1 else -self.f_mag
        u = action*self.f_mag

        mass = self.m_c+self.m_p #total mass
        pm = self.m_p*self.l # polemass
        cart_friction = self.mu_c*np.sign(x_d)
        pole_friction = (self.mu_p*theta_d)/pm
        factor = (u + pm*np.square(theta_d)*np.sin(theta) - cart_friction)/mass

        theta_dd_num = (self.g*np.sin(theta) - np.cos(theta)*factor
                        - pole_friction)
        theta_dd_den = self.l*(4.0/3-(self.m_p*np.square(np.cos(theta)))/mass)
        theta_dd = theta_dd_num/theta_dd_den

        x_dd = factor - pm*theta_dd*np.cos(theta)/mass

        return state + np.array([x_d, x_dd, theta_d, theta_dd])*self.dt

class AMPPI:
    """ Implements an MMPI controller as defined in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning', for the
    OpenAI cart-pole environment.

    Keyword arguments:
    env -- an OpenAi Gym environment
    K -- number of sampled trajectories
    T -- number of timesteps in the control horizon
    U -- control matrix of the form [u_0, ..., u_T]
    lambda_ --
    u_init -- guess for initial control action
    gaussian_noise -- bool which controls if Gaussian noise is added to the
    control outputs
    epsilon_loc -- mean of the Gaussian noise
    epsilon_scale -- variance of the Gaussian noise
    """

    def __init__(self, obs_space, act_space, K, T, lambda_=1.0,
                 gaussian_noise=True, epsilon_loc=0, epsilon_scale=1):
        self.K = K
        self.T = T
        self.lambda_ = lambda_
        self.m = act_space.shape[0] if act_space.shape else 1 # 1 if Discrete
        self.n = obs_space.shape[0] if obs_space.shape else 1
        self.gaussian_noise = gaussian_noise
        self.epsilon_loc = epsilon_loc
        self.epsilon_scale = epsilon_scale
        self.U = np.zeros((self.m, T))
        self.H = np.eye(self.m)  # TODO Figure out proper value


    def compute_cost(self, state):
        return (10*np.square(state[0]) + 500*(np.cos(state[2]) + 1) +
    np.square(state[1]) + 15*np.square(state[3]))


    def compute_trajectory_cost(self, model, state, epsilon):
        """ Computes the total cost of a single control trajectory
        Keyword arguments:
        model -- a model of the CartPole class, holds the last estimate of
        system dynamics
        state -- the trajectory initial state
        epsilon -- a sampled noise vector for the complete trajectory
        """
        S = 0
        for t in range(self.T):
            # Takes a step in the dynamical model and sets new state
            state = model.step(state, self.U[:, t]+epsilon[:, t])
            S += (self.compute_cost(state)
                  + self.lambda_*self.U[:, t].T*self.H*epsilon[:, t])
        # S += terminal_cost(state) -- lets disregard the terminal cost for now
        return S

    def sample_trajectories(self, model, state):
        if self.gaussian_noise:
            epsilon = np.random.normal(loc=self.epsilon_loc,
                                       scale=self.epsilon_scale,
                                       size=(self.K, self.m, self.T))
        else:
            epsilon = np.zeros(shape=(m, self.T))

        cost = np.zeros(self.K)
        eta = 0
        for k in range(self.K):
            cost[k] = self.compute_trajectory_cost(model, state, epsilon[k])
        beta = np.min(cost)
        eta = np.sum(np.exp(-1/self.lambda_*(cost-beta)))  # returns a scalar
        omega = np.exp(-1/self.lambda_*(cost-beta)/eta)  # vector of length k
        return epsilon, omega, beta

    def act(self, model, state):
        epsilon, omega, beta = self.sample_trajectories(model, state)
        for t in range(self.T):
            self.U[:, t] = np.dot(omega.T, epsilon[:, :, t])
        self.U = (self.U>=0).astype(int)  # Binarizes actions
        action = self.U[:, 0]
        self.U = np.roll(self.U, -1, axis=1)  # Shifts U_t+1 to U_t
        self.U[:, self.T-1] = np.zeros_like(self.U[:, self.T-1])
        return action, beta


def loop():
    env = gym.make(ENV_NAME)
    state = env.reset()
    model = CartPoleModel()
    controller = AMPPI(env.observation_space, env.action_space, N_SAMPLES,
                       TIMESTEPS)
    step = 0
    while True:
        step += 1
        env.render()
        action, cost = controller.act(model, state)
        action = int(action)  # converts from np.array to scalar
        state, _, terminal, _ = env.step(action)
        if not step%10:
            print("Step {0}: forecast cost {1}".format(step, cost))
            print("Next actions: {}".format(controller.U))
        if terminal:
            env.close()
            break


if __name__ == "__main__":
    loop()
