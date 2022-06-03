from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion


def convert(raw, bias, sens):
	value = (raw - bias) * 3300/(1023 * sens) * 9.81
	return value
	
data_num = 1
imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
vicon = io.loadmat('source/vicon/viconRot'+str(data_num)+'.mat')
accel = imu['vals'][0:3,:]
gyro = imu['vals'][3:6,:]
T = np.shape(imu['ts'])[1]


bias_x = np.round(np.average(gyro[1,0:100]))
bias_y = np.round(np.average(gyro[2,0:100]))
bias_z = np.round(np.average(gyro[0,0:100]))

print(bias_x)
print(bias_y)
print(bias_z)

plt.plot(gyro[1,:], label="gyroX")
plt.plot(gyro[2,:], label="gyroY")
plt.plot(gyro[0,:], label="gyroZ")
plt.legend()
plt.show()
