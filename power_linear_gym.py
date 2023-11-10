import gym
from gym import spaces
from EagarTsaiModel import EagarTsai as ET
import numpy as np
import os
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from pylab import gca

# This script implements the RL environment as a custom OpenAI Gym environment, using Stable Baselines 3

# Dependencies: Stable Baselines 3, OpenAI Gym, Pytorch

# Specify local figure directory to store plots, diagnostic info

fig_dir = 'results/linear_power_control_figures'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)


def frame_tick(frame_width=2, tick_width=1.5):
    ax = gca()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)

    plt.tick_params(direction='in',

                    width=tick_width)


class EnvRLAM(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, plot=False, frameskip=1, num_iter=0, verbose=0):
        super(EnvRLAM, self).__init__()  # Define action and observation space
        # They must be gym.spaces objects
        self.plot = plot
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)
        self.squaresize = 20
        self.spacing = 20e-3
        self.observation_space = spaces.Box(low=300, high=20000, shape=(9, self.squaresize, self.squaresize,),
                                            dtype=np.float64)
        self.ETenv = ET(20e-3, V=0.8, bc='flux', spacing=self.spacing)
        self.current_step = 0
        self.depths = []
        self.times = []
        self.indtimes = []
        self.inddepth = []
        self.indpower = []
        self.indvel = []
        self.power = []
        self.velocity = []
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.dt = 0
        self.distance = 0
        self.dir = 0
        self.residual = False
        self.reward = 0
        self.timesteps = 0
        self.num_iter = num_iter
        self.frameskip = frameskip
        self.verbose = verbose
        self.total_steps = 0
        buffer = np.stack((self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                           self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))
        # buffer to be using for loop to reduce to one sentence

        self.buffer = (buffer - np.mean(buffer))

    def step(self, action):

        time = [0.00397, 0.00983, 0.01205,0.01594, 0.01997, 0.02395, 0.02791, 0.03196, 0.0359]
        power = action[0] * 250 + 250
        V = [0.01953078, 0.01974113, 0.02015934, 0.0135428, 0.03482298, 0.01917126, 0.0198257, 0.02003466, 0.02078936, 0.02000099]

        for m in range(self.frameskip):
            done = False
            self.current_step += 1
            self.velocity.append(V)
            self.power.append(power)
            idx = self.current_step - 1
            if idx < 0:
                idx = 0

            angle = 0
            self.dir = 'right'
            total_distance = 1250e-3 * 0.8 - 125e-3

            if (self.distance < total_distance):
                self.distance += V[self.current_step - 1] * time[self.current_step - 1]
                self.timesteps += 1

            self.ETenv.forward(time[self.current_step-1], angle, V=V[self.current_step-1], P=power)

            meltpool = self.ETenv.meltpool()
            self.depths.append(meltpool)
            self.times.append(self.ETenv.time)

            if m == 0:
                self.inddepth.append(meltpool)
                self.indtimes.append(self.ETenv.time)
                self.indvel.append(V)
                self.indpower.append(power)

            reward = 1 - np.abs((55e-3 + meltpool) / 25e-3)

            self.reward += reward

            # Plotting diagnostics

            if self.plot:
                np.savetxt(fig_dir + "/" + "powercontrollineartimesnorm", np.array(self.times))
                np.savetxt(fig_dir + "/" + "powercontrollineardepthsframeskip" + str(self.frameskip),
                           np.array(self.depths))
                np.savetxt(fig_dir + "/" + "powercontrollinearvelocityframeskip" + str(self.frameskip),
                           np.array(self.velocity))
                testfigs = self.ETenv.plot()
                highxlim = np.max(self.times)
                testfigs[0].savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollinear_test' + '%04d' % self.current_step + ".png")
                plt.clf()

                font_size = 14
                plt.plot(np.array(self.times), np.array(self.depths) * 1e3, linewidth=2.0)
                plt.ylim(-120, 10)
                plt.xlabel(r'Time, $t$ [s]', fontsize=font_size)
                plt.ylabel(r'Melt Depth, $d$, [$mm]', fontsize=font_size)
                plt.plot(np.array(self.indtimes), np.array(self.inddepth) * 1e3, 'k.')
                np.max(np.array(self.times))
                plt.xlim(0, highxlim)
                plt.title(str(round(self.ETenv.time)) + r'[$s] ')
                plt.plot(np.arange(0, np.max(np.array(self.times)), 0.01),
                         -55 * np.ones(len(np.arange(0, np.max(np.array(self.times)), 0.01))), 'k--')
                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestdepth' + '%04d' % self.current_step + ".png")

                plt.clf()
                plt.plot(np.array(self.times), self.velocity)
                plt.plot(np.array(self.indtimes), np.array(self.indvel), 'k.', linewidth=2.0)
                plt.xlabel(r'Time, $t$ [ms]', fontsize=font_size)
                plt.ylabel(r'Velocity, $V$, [mm/s]', fontsize=font_size)
                plt.xlim(0, highxlim)
                plt.ylim(0, 3.0)
                plt.title(str(round(self.ETenv.time)) + r'[$s] ')
                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestvelocity' + '%04d' % self.current_step + ".png")

                plt.clf()
                plt.plot(np.array(self.times), self.power)
                plt.plot(np.array(self.indtimes), np.array(self.indpower), 'k.', linewidth=2.0)
                plt.xlabel(r'Time, $t$ [ms]', fontsize=font_size)
                plt.ylabel(r'Power, $P$, [W]', fontsize=font_size)
                plt.xlim(0, highxlim)
                plt.ylim(-10, 600)
                plt.title(str(round(self.ETenv.time)) + r'[$s] ')
                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestpower' + '%04d' % self.current_step + ".png")

                plt.clf()

            idxx = self.ETenv.location_idx[0]
            idxy = self.ETenv.location_idx[1]
            self.buffer[8, :, :] = np.copy(self.buffer[5, :, :])
            self.buffer[7, :, :] = np.copy(self.buffer[4, :, :])
            self.buffer[6, :, :] = np.copy(self.buffer[3, :, :])
            self.buffer[5, :, :] = np.copy(self.buffer[2, :, :])
            self.buffer[4, :, :] = np.copy(self.buffer[1, :, :])
            self.buffer[3, :, :] = np.copy(self.buffer[0, :, :])

            padsize = self.squaresize

            theta_pad = np.copy(
                np.pad(self.ETenv.theta, ((padsize, padsize), (padsize, padsize), (0, 0)), mode='reflect'))

            try:

                self.buffer[2, :, :] = theta_pad[
                                       padsize + idxx - self.squaresize // 2:padsize + idxx + self.squaresize // 2,
                                       padsize + idxy - self.squaresize // 2:padsize + idxy + self.squaresize // 2, -1]

                self.buffer[1, :, :] = theta_pad[
                                       padsize + idxx - self.squaresize // 2:padsize + idxx + self.squaresize // 2,
                                       padsize + idxy, 0:self.squaresize]

                self.buffer[0, :, :] = theta_pad[padsize + idxx,
                                       padsize + idxy - self.squaresize // 2:padsize + idxy + self.squaresize // 2,
                                       0:self.squaresize]

            except:

                breakpoint()

            self.buffer[0:3] = (self.buffer[0:3] - np.mean(self.buffer[0:3], axis=(1, 2))[:, None, None]) / (
                    np.std(self.buffer[0:3], axis=(1, 2))[:, None, None] + 1e-10)

            obs = self.buffer

            if self.timesteps > 9:
                minmax_reward = (np.max(self.depths[10:]) - np.min(self.depths[10:])) / 25e-3
                gradient_reward = np.mean(np.abs(np.diff(self.depths[10:]) / 25e-3))
                reward = self.reward/10 - 0.5 * minmax_reward

                if self.verbose == 2:
                    print(reward, "reward", "velocity", V, "power", power, "timestep", self.total_steps, "amplitude",
                          minmax_reward)

                    print("Episode done")

                if self.verbose == 1:
                    if self.total_steps % 1 == 0:
                        print(str(self.total_steps * 8) + " episodes run, current reward = " + str(reward))

                if self.verbose == 0:
                    if self.total_steps % 10 == 0:
                        print(str(self.total_steps * 8) + " episodes run, current reward = " + str(reward))

                done = True
                self.total_steps += 1



            else:
                done = False
            if done:
                break

        return obs, reward, done, {}

    # Execute one time step within the environment

    def reset(self):
        self.ETenv.reset()
        self.current_step = 0
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.residual = False
        self.dir = 0
        step = 2
        self.reward = 0
        self.timesteps = 0
        self.dt = 0
        self.distance = 0
        self.buffer = np.stack((self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))

        buffer = np.stack((

            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],

            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
            self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))

        self.buffer = (buffer - np.mean(buffer))

        return buffer

    # Reset the state of the environment to an initial state

    def render(self, mode='console', close=False):
        pass

    def plot(self):
        pass

    def plot_buffer(self, action):
        time = action[0] * 125 + 140
        power = 145
        V = 20e-3 / time
        buffer_dir = fig_dir + '/buffer/'
        if not os.path.exists(buffer_dir):
            os.makedirs(buffer_dir)
        for index in range(9):
            plt.close('all')
            try:
                if index % 3 == 2:
                    plt.pcolormesh(self.buffer[index].T, cmap='jet')
                if index % 3 == 1:
                    plt.pcolormesh(self.buffer[index].T, cmap='jet')
                if index % 3 == 0:
                    plt.pcolormesh(self.buffer[index].T, cmap='jet')
                # breakpoint()

            except Exception as e:
                print(e)
                breakpoint()
                print('saved')

            plt.title("time: " + str(V) + " power: " + str(power))
            plt.savefig(buffer_dir + str(index) + "closelinearbuffer" + '%04d' % self.current_step + ".png")
            plt.clf()
            plt.close('all')


def main():
    env = EnvRLAM()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
