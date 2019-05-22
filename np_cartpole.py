import gym
import numpy as np
import np_control

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
        x, x_d, theta, theta_d = np.array(state)
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
        delta = np.array([x_d, x_dd, theta_d, theta_dd])*self.dt
        return state+delta

    def compute_terminal_cost(self, state):
        terminal = state[0] < -2.4 \
                   or state[0] > 2.4 \
                   or state[2] < -12*2*np.pi/360 \
                   or state[2] > 12*2*np.pi/360
        return 1000000 if terminal else 0

    def compute_state_cost(self, state):
        # Original cost function on MPPI paper
        return (1*np.square(state[0]) + 500*np.square(np.sin(state[2]))
                + 1*np.square(state[1]) + 1*np.square(state[3])).reshape(-1, 1)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state = env.reset()
    state = np.array(state).reshape(-1, 1)  # Reshape to a Numpy row vector
    model = CartPoleModel(mu_p=0, mu_c=0)
    controller = np_control.MPPI(obs_space=env.observation_space,
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
                  np.around(controller.U[:4].T, 2)))
        state, _, done, _ = env.step(action)
        state = np.array(state).reshape(-1, 1)  # Reshape to a Numpy row vector
        # state = model.step(state, action)
        # terminal = state[0] < -2.4 \
        #            or state[0] > 2.4 \
        #            or state[2] < -12*2*np.pi/360 \
        #            or state[2] > 12*2*np.pi/360
        # terminal = bool(terminal)
        if done:
            print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Last state: x={0[0]}, theta={0[2]}".format(state))
            print("Next actions: {}".format(np.around(controller.U.T, 2)))
            env.close()
            break
        step += 1
