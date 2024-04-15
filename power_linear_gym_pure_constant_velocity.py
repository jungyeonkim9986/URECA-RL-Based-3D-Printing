from EagarTsaiModel import EagarTsai as ET
import numpy as np
import os
from matplotlib import pyplot as plt
from pylab import gca

fig_dir = 'results/DED_constant_V_linear_power_control_figures'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)


def frame_tick(frame_width=2, tick_width=1.5):
    ax = gca()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)

    plt.tick_params(direction='in',

                    width=tick_width)


class MeltPoolSimulation:
    def __init__(self, frameskip=1, plot=False, fig_dir="results"):
        self.frameskip = frameskip
        self.current_step = 0
        self.indtimes = []
        self.inddepth = []
        self.indpower = []
        self.indvel = []
        self.squaresize = 10
        self.angle = 0
        self.spacing = 10e-5
        self.velocity = []
        self.power = []
        self.dir = 0
        self.depths = []
        self.times = []
        self.distance = 0
        self.plot = plot
        self.fig_dir = fig_dir
        self.ETenv = ET(10e-5, V=0.020, bc='flux',
                        spacing=self.spacing)  # Instance of the Eagar-Tsai model for melt pool calculations

    def step(self):
        # Time and velocity values from real-world printing
        times = [0.00001, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
                 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64,
                 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
                 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32,
                 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66,
                 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00,
                 2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.22, 2.24, 2.26, 2.28, 2.30, 2.32, 2.34,
                 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.54, 2.56, 2.58, 2.60]
            # , 2.62, 2.64, 2.66, 2.68,
            #      2.70, 2.72, 2.74, 2.76, 2.78, 2.80, 2.82, 2.84, 2.86, 2.88, 2.90, 2.92, 2.94, 2.96, 2.98, 3.00, 3.02,
            #      3.04, 3.06, 3.08, 3.10, 3.12, 3.14, 3.16, 3.18, 3.20, 3.22, 3.24, 3.26, 3.28, 3.30, 3.32, 3.34, 3.36,
            #      3.38, 3.40]
        # times = [0.00001, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
        #          3.4, 3.6, 3.8]
        power = 2300  # Fixed power value for pure simulation
        velocity = [0.025] * 131
        # velocity = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        #             0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]

        for m in range(len(times)):
            self.current_step += 1
            self.velocity.append(velocity[self.current_step - 1])
            self.power.append(power)
            self.angle = 0
            self.dir = 'right'

            if self.current_step <= len(times):
                if self.current_step == 1:
                    time = times[self.current_step - 1]
                    self.distance += velocity[self.current_step - 1] * time
                    self.ETenv.forward(time, 0, V=velocity[self.current_step - 1], P=power)
                else:
                    time = times[self.current_step - 1] - times[self.current_step - 2]
                    self.distance += velocity[self.current_step - 1] * time
                    self.ETenv.forward(time, 0, V=velocity[self.current_step - 1], P=power)

            meltpool = self.ETenv.meltpool()

            self.depths.append(meltpool)
            self.times.append(self.ETenv.time)

            self.inddepth.append(meltpool)
            self.indtimes.append(times[self.current_step - 1])
            self.indvel.append(velocity[self.current_step - 1])
            self.indpower.append(power)


            # Plotting diagnostics
            if self.plot:
                np.savetxt(fig_dir + "/" + "powercontrollineartimesnorm", np.array(self.times) * 1e3)
                np.savetxt(fig_dir + "/" + "powercontrollineardepthsframeskip" + str(self.frameskip),
                           np.array(self.depths))

                np.savetxt(fig_dir + "/" + "powercontrollinearvelocityframeskip" + str(self.frameskip),
                           np.array(self.velocity))
                testfigs = self.ETenv.plot()
                highxlim = np.max(self.times)
                testfigs[0].savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollinear_test' + '%04d' % self.current_step + ".png")
                plt.close()
                plt.clf()
                font_size = 14
                plt.plot(np.array(self.times), np.array(self.depths) * 1e3, linewidth=2.0)
                plt.ylim(-1.3, 0)
                plt.xlabel(r'Time, $t$ [s]', fontsize=font_size)
                plt.ylabel(r'Melt Depth, $d$, [mm]', fontsize=font_size)
                plt.plot(np.array(self.indtimes), np.array(self.inddepth) * 1e3)
                np.max(np.array(self.times))
                plt.xlim(0, highxlim)
                plt.title(str(round(self.ETenv.time)) + r'[s] ')

                # plt.plot(np.arange(0, np.max(np.array(self.times)), 0.01),
                #          -2000 * np.ones(len(np.arange(0, np.max(np.array(self.times)) * 1e3, 0.01))), 'k--')

                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestdepth' + '%04d' % self.current_step + ".png")
                plt.close()
                plt.clf()

                plt.plot(np.array(self.times), self.velocity)
                plt.plot(np.array(self.indtimes), np.array(self.indvel) * 1e3, linewidth=2.0)
                plt.xlabel(r'Time, $t$ [ms]', fontsize=font_size)
                plt.ylabel(r'Velocity, $V$, [mm/s]', fontsize=font_size)
                plt.xlim(0, highxlim)
                plt.ylim(0, 40)
                plt.title(str(round(self.ETenv.time)) + r'[s] ')
                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestvelocity' + '%04d' % self.current_step + ".png")
                plt.close()
                plt.clf()

                plt.plot(np.array(self.times), self.power)
                plt.plot(np.array(self.indtimes), np.array(self.indpower), linewidth=2.0)
                plt.xlabel(r'Time, $t$ [s]', fontsize=font_size)
                plt.ylabel(r'Power, $P$, [W]', fontsize=font_size)
                plt.xlim(0, highxlim)
                plt.ylim(2000, 2500)
                plt.title(str(round(self.ETenv.time)) + r'[s] ')
                plt.savefig(fig_dir + "/" + str(
                    self.frameskip) + 'powercontrollineartestpower' + '%04d' % self.current_step + ".png")
                plt.close()
                plt.clf()

    # Additional methods for plotting and other functionalities


# Example usage
if __name__ == '__main__':
    sim = MeltPoolSimulation(plot=True)
    sim.step()
