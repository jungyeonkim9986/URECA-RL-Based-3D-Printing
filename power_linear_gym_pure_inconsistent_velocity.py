from EagarTsaiModel import EagarTsai as ET
import numpy as np
import os
from matplotlib import pyplot as plt
from pylab import gca
import pandas as pd

fig_dir = 'results/DED_inconsistent_V_linear_power_control_figures'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

def frame_tick(frame_width=2, tick_width=1.5):
    ax = gca()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(frame_width)

    plt.tick_params(direction='in',

                    width=tick_width)

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def find_closest_velocity(time, df):
    closest_index = df['Time_1'].sub(time).abs().idxmin()
    return df.iloc[closest_index]['field.Vx']

def get_velocities_for_times(times_list, df):
    velocities = []
    for time in times_list:
        velocity = find_closest_velocity(time, df)
        velocity_m_s = velocity/1000
        velocities.append(velocity_m_s)
    return velocities

def get_velocity_list(csv_file_path, times_list):
    df = load_csv_data(csv_file_path)
    return get_velocities_for_times(times_list, df)

class MeltPoolSimulation:
    def __init__(self, frameskip=1, plot=False, fig_dir="results"):
        self.frameskip = frameskip
        self.current_step = 0
        self.indtimes = []
        self.inddepth = []
        self.indpower = []
        self.indvel = []
        self.squaresize = 20
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
        self.ETenv = ET(10e-5, V=0.025, bc='flux', spacing=self.spacing)   # Instance of the Eagar-Tsai model for melt pool calculations

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
        power = 2300  # Fixed power value for pure simulation
        csv_file_path = r'C:\Users\willi\PycharmProjects\pythonProject\URECA-RL-Based-3D-Printing/position_21_raw.csv'
        velocity = get_velocity_list(csv_file_path, times)

        for index, vel in enumerate(velocity):
            print(f"Index {index}: Velocity {vel}")
        # velocity = [0.019530785, 0.019171261, 0.019944061, 0.019286718, 0.019365482, 0.019508972, 0.019412291,
        #             0.020662823, 0.020581358, 0.019672026, 0.020573168, 0.020168806, 0.020031038, 0.019268173,
        #             0.019892824, 0.020198988, 0.020207396, 0.020148415, 0.020189665, 0.020695328, 0.020365034,
        #             0.019606339, 0.019875341, 0.020209896, 0.019488834, 0.019976461, 0.020038183, 0.019214294,
        #             0.019964516, 0.020510347, 0.020088285, 0.02091641, 0.020244192, 0.020266491, 0.019748858,
        #             0.020376259, 0.020998476, 0.020237654, 0.019678471, 0.019557142, 0.019506643, 0.021918638,
        #             0.020447973, 0.019830093, 0.019736155, 0.019497635, 0.021212151, 0.019511847, 0.019318495,
        #             0.018976929, 0.020624842, 0.019022568, 0.019859726, 0.02005571, 0.019561548, 0.020053541,
        #             0.020019003, 0.01949173, 0.021060761, 0.019964516, 0.019604343, 0.019803585, 0.020029335,
        #             0.020082771, 0.016427431, 0.020156158, 0.020032497, 0.020840174, 0.020696424, 0.01976955,
        #             0.019288208, 0.020153999, 0.019643791, 0.019783976, 0.020074005, 0.020030577, 0.020177307,
        #             0.020267912, 0.019358704, 0.020829044, 0.020221685, 0.019550659, 0.020677219, 0.01984267,
        #             0.019315561, 0.019645277, 0.019826437, 0.019569407, 0.020518385, 0.019897411, 0.020382343,
        #             0.020096777, 0.021091972, 0.019493546, 0.019818361, 0.019739683, 0.018921303, 0.019368721,
        #             0.020318253, 0.019421949, 0.01962582, 0.019876562, 0.019665089, 0.020596916, 0.019555801,
        #             0.02005649, 0.019843042, 0.019752129, 0.019759876, 0.020002716, 0.020006516, 0.019766682,
        #             0.019947594, 0.02045451, 0.019739925, 0.020243658, 0.020675793, 0.020059069, 0.020052048,
        #             0.01985663, 0.019520292, 0.019885719, 0.020351629, 0.019123484, 0.020901598, 0.020062105,
        #             0.020321768, 0.019957258, 0.01984543, 0.02005324, 0.020374723, 0.02064674, 0.019247564,
        #             0.019674881, 0.020361677, 0.019664253, 0.019422966, 0.020115719, 0.019901154, 0.021203875,
        #             0.019712458, 0.019634274, 0.020385166, 0.020792753, 0.019384506, 0.019426729, 0.01950021,
        #             0.019891321, 0.020468584, 0.019731169, 0.01992239, 0.020153864, 0.020757139, 0.019211689,
        #             0.019503716, 0.01973908, 0.020057463, 0.019916382, 0.019743567, 0.020252201, 0.020716354,
        #             0.02033812, 0.019604553, 0.020303991, 0.017124172, 0.016740015, 0.012118902, 0.009146301,
        #             0.006662274, 0.003525904, 0.001322715]
        self.dir = 'right'

        for m in range(len(times)):
            self.angle = 0
            self.current_step += 1
            self.velocity.append(velocity[self.current_step-1])
            self.power.append(power)

            if self.current_step <= len(times):
                if self.current_step == 1:
                    time = times[self.current_step-1]
                    self.distance += velocity[self.current_step - 1] * time
                    self.ETenv.forward(time, 0, V=velocity[self.current_step - 1], P=power)
                else:
                    time = times[self.current_step-1]-times[self.current_step-2]
                    self.distance += velocity[self.current_step - 1] * time
                    self.ETenv.forward(time, 0, V=velocity[self.current_step - 1], P=power)


            meltpool = self.ETenv.meltpool()
            
            self.depths.append(meltpool)
            self.times.append(self.ETenv.time)

            self.inddepth.append(meltpool)
            self.indtimes.append(times[self.current_step-1])
            self.indvel.append(velocity[self.current_step-1])
            self.indpower.append(power)

            self.ETenv.get_coords()

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
                plt.xlabel(r'Time, $t$ [s]', fontsize=font_size)
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

# Example usage
if __name__ == '__main__':
    sim = MeltPoolSimulation(plot=True)
    sim.step()
