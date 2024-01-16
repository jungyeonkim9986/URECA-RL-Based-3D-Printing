from EagarTsaiModel import EagarTsai as ET
import numpy as np
import os
from matplotlib import pyplot as plt
from pylab import gca

fig_dir = 'results/linear_power_control_figures'

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
        self.velocity = []
        self.power = []
        self.depths = []
        self.times = []
        self.distance = 0
        self.plot = plot
        self.fig_dir = fig_dir
        self.ETenv = ET()   # Instance of the EagarTsai model for melt pool calculations

    def step(self):
        # Time and velocity values from real-world printing
        time = [0, 0.00397, 0.00983, 0.01205, 0.01594, 0.01997, 0.02395, 0.02791, 0.03196, 0.0359]
        power = 1450  # Fixed power value for pure simulation
        V = [0.01953078, 0.01974113, 0.02015934, 0.0135428, 0.03482298, 0.01917126, 0.0198257, 0.02003466, 0.02078936, 0.02000099]

        for m in range(len(time)):
            self.current_step += 1
            self.velocity.append(V[m])
            self.power.append(power)

            if self.current_step < len(time):
                self.distance += V[self.current_step - 1] * time[self.current_step - 1]

            self.ETenv.forward(time[self.current_step-1], 0, V=V[self.current_step-1], P=power)

            meltpool = self.ETenv.meltpool()
            self.depths.append(meltpool)
            self.times.append(self.ETenv.time)

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
                         -2 * np.ones(len(np.arange(0, np.max(np.array(self.times)), 0.01))), 'k--')
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


    # Additional methods for plotting and other functionalities

# Example usage
if __name__ == '__main__':
    sim = MeltPoolSimulation(plot=True)
    sim.step()
    # Further processing or analysis can be done based on the simulation results
