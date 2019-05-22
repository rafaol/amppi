import gym
import numpy as np
import np_control

ENV_NAME = "Pendulum-v0"
TIMESTEPS = 20   # T
N_SAMPLES = 500  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
LAMBDA_ = 1
EPS_SCALE = 10

class PendulumModel:
    """For information refer to OpenAI Gym environment at:
    https://gym.openai.com/envs/Pendulum-v0/
    """
    def __init__(self, g=9.8, m=1.0, l=1.0, max_torque=2., max_speed=8,
                 dt=0.05):
        self.g = g  #m/s^2
        self.m = m  #kg
        self.l = l  #m
        self.max_torque = max_torque  # N/m
        self.max_speed = max_speed  # m/s
        self.dt = dt #s

    def step(self, state, action):
        theta, theta_d = state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        theta_d = theta_d + dt*(-3*g/(2*l) * np.sin(theta+np.pi)
                                + 3./(m*l**2)*action)
        theta = theta + theta_d*dt  # Use new theta_d
        theta_d = np.clip(theta_d, -self.max_speed, self.max_speed)
        return np.array([theta, theta_d])

    def compute_terminal_cost(self, state):
        return 0

    def compute_state_cost(self, state):
        # Note that theta may range beyond 2*np.pi
        theta, theta_d = state
        return (10*(np.cos(theta)-1)**2 + .1*theta_d**2).reshape(-1, 1)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    env.reset()
    state = env.state
    state = np.array(state).reshape(-1, 1)  # Reshape to a Numpy row vector
    model = PendulumModel()
    controller = np_control.MPPI(obs_space=env.observation_space,
                                 act_space=env.action_space,
                                 K=N_SAMPLES,
                                 T=TIMESTEPS,
                                 lambda_=LAMBDA_,
                                 eps_scale=EPS_SCALE)
    step = 0
    while step<200:
        env.render()
        action, tau, cost, omega = controller.act(model, state)
        if not step%25:
            print("Step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Current state: "
                  "theta={0[0]}, theta_dot={0[1]}".format(state))
            print("Next actions 4 actions: {}".format(
                  np.around(controller.U[:4].T, 2)))
        # input("Press Enter to continue...")
        _, _, done, _ = env.step(action)
        state = env.state
        state = state.reshape(-1, 1)  # Reshape to a Numpy row vector
        if done:
            print("Last step {0}: forecast cost {1:.2f}".format(step, cost))
            print("Last state: "
                  "theta={0[0]}, theta_dot={0[1]}".format(state))
            print("Next actions: {}".format(np.around(controller.U.T, 2)))
            env.close()
            break
        step += 1
