import numpy as np
import matplotlib.pyplot as plt

times = [0.00001, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
         0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64,
         0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
         1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32,
         1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66,
         1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00,
         2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.22, 2.24, 2.26, 2.28, 2.30, 2.32, 2.34,
         2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.54, 2.56, 2.58, 2.60]

inconsistent_velocity = np.loadtxt('C:/Users/willi/PycharmProjects/pythonProject/URECA-RL-Based-3D-Printing/'
                                   'inconsistent_velocity_frameskip.txt') * 1000  # Convert to mm/s

constant_velocity = np.full(shape=(131,), fill_value=25)  # 25 mm/s for each time stamp

plt.figure(figsize=(10, 6))
plt.plot(times, constant_velocity, label='Constant Velocity', color='blue')
plt.plot(times, inconsistent_velocity, label='Inconsistent Velocity', color='orange')

plt.xlim(left=0)  # Set the x-axis to start from the minimum time value
plt.ylim(bottom=0)         # Set the y-axis to start from 0

plt.xlabel('Time, t [s]', fontsize = 20)
plt.ylabel('Velocity, V [mm/s]', fontsize = 20)
# plt.title('Comparison of Constant and Inconsistent Velocity Profiles')
plt.legend(loc='lower right')

plt.savefig(r'C:\Users\willi\PycharmProjects\pythonProject\URECA-RL-Based-3D-Printing')

plt.show()
