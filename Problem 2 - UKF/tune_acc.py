from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion


def convert(raw, bias, sens):
	value = (raw - bias) * 3300/(1023 * sens) * 9.81
	return value
	
data_num = 1
imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
accel = imu['vals'][0:3,:]
gyro = imu['vals'][3:6,:]
T = np.shape(imu['ts'])[1]


bias_x = np.round(np.average(accel[0,0:100]))
bias_y = np.round(np.average(accel[1,0:100]))


print(bias_x)
print(bias_y)
z = []

quat = Quaternion()
for i in range(3000,3500):
	rot = vicon['rots'][:,:,i].reshape(3,3)
	q = quat.from_rotm(rot)
	euler_angles = quat.euler_angles()
	if(euler_angles[0] >= np.pi/2):
		z.append(accel[2,i])
	
bias_z = np.round(np.average(z))
print(bias_z)


raw_z = accel[2,0:100]
alpha =  np.round(np.average((raw_z - bias_z) * 3300/1023))
print(alpha)

roll = []
pitch = []
yaw = []
for i in range(vicon['rots'].shape[-1]):
	rot = vicon['rots'][:,:,i].reshape(3,3)
	q = quat.from_rotm(rot)
	euler_angles = quat.euler_angles()
	roll.append(euler_angles[0])
	pitch.append(euler_angles[1])
	yaw.append(euler_angles[2])

#plt.plot(roll, label="vicon roll")
plt.plot(pitch, label="vicon pitch")


accel_x = convert(accel[0,:], bias_x, alpha)
accel_y = convert(accel[1,:], bias_y, alpha)
accel_z = convert(accel[2,:], bias_z, alpha)

roll = np.arctan(-accel_y/accel_z)
pitch = np.arctan(accel_x/np.sqrt(accel_y**2 + accel_z**2))

#plt.plot(roll, label="roll calc")
plt.plot(pitch, label="pitch calc")
plt.legend()
plt.show()
