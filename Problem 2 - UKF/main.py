import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def digitalToAnalog(raw, sens, bias):
    val = (raw - bias) * 3300 / (1023 * sens)
    return val
    
def generate_sigma(x_k_1, P_k_1, Q):
    S = np.linalg.cholesky((P_k_1 + Q)*np.sqrt(6))
    W = np.hstack((S, -S))
    
    q_x = Quaternion(x_k_1[0,0], x_k_1[1:4,0])
    q_x.normalize()
    
    X = np.zeros((7,12))
    
    for i in range(12):
        q_w = Quaternion()
        q_w.from_axis_angle(W[0:3,i])
        q_X = q_x * q_w
        q_X.normalize()
        
        X[0:4, i] = q_X.q
        X[4:7, i] = x_k_1[4:7,0] + W[3:6, i]
    
    return X
    
    
def transform(X, delta_t):

    for i in range(12):
        q_delta = Quaternion()
        q_delta.from_axis_angle(X[4:7, i] * delta_t)
        q_X = Quaternion(X[0,i], X[1:4,i])
        q_Y = q_X * q_delta
        q_Y.normalize()
        X[0:4,i] = q_Y.q

    return X

def get_mean_q(q_Y, q_init):
    qt = Quaternion(q_init[0], q_init[1:4])
    prev_e = 10
    threshould = 0.0001
    count = 0
    e_avg = np.array([0,0,0])
    while(np.abs(np.linalg.norm(e_avg) -  prev_e) > threshould and count < 50):
        prev_e = np.linalg.norm(e_avg)

        e_i = np.zeros((3, 12))
        for i in range(12):
            q_Yi =  Quaternion(q_Y[0,i], q_Y[1:4,i])
            r_w = q_Yi * qt.inv()
            r_w.normalize()
            e_i[:, i] = r_w.axis_angle()

        e_avg = np.mean(e_i, axis=1)

        q_e = Quaternion()
        q_e.from_axis_angle(e_avg)
        qt = q_e * qt
        qt.normalize()

        count += 1
    return qt

def get_mean(Y, q_init):

    x_k_bar = np.zeros((7,1))
    x_k_bar[4:7,0] = np.mean(Y[4:7,:], axis = 1)
    q_mean_Y = get_mean_q(Y[0:4,:], q_init)
    x_k_bar[0:4,0] = q_mean_Y.q
    
    return x_k_bar
    
    
def get_W(Y, x_k_bar):

    W = np.zeros((6,12))
    q_mean = Quaternion(x_k_bar[0,0], x_k_bar[1:4,0])
    omega_mean = x_k_bar[4:7,0]

    for i in range(12):
        q_Y = Quaternion(Y[0,i], Y[1:4,i])
        mul = q_Y * q_mean.inv()
        mul.normalize()
        r_W = mul.axis_angle()
        w_W = Y[4:7,i] - omega_mean
        W[:,i] = np.hstack((r_W, w_W))

    return W
        

def get_Z(Y):
    Z = np.zeros((6,12))
    g = Quaternion(0.0, [0.0, 0.0, 9.8])

    for i in range(12):
        q_Yi = Quaternion(Y[0,i], Y[1:4,i])
        Z[0:3, i] = (q_Yi.inv() * g * q_Yi).vec() 
        Z[3:6, i] = Y[4:7, i]    
    
    return Z



def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('source/vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here
    ax = -digitalToAnalog(accel[0, :], 33.86, 511.0)
    ay = -digitalToAnalog(accel[1, :], 33.86, 501.0)
    az = digitalToAnalog(accel[2, :], 33.86, 503.0)

    gx = digitalToAnalog(gyro[1, :], 193.55, 377.0)
    gy = digitalToAnalog(gyro[2, :], 193.55, 371.5)
    gz = digitalToAnalog(gyro[0, :], 193.55, 369.5) 


    delta_t = imu['ts'][0, 1:] - imu['ts'][0, 0:-1]
    delta_t = np.hstack((0,delta_t))

	
    vicon_roll = []
    vicon_pitch = []
    vicon_yaw = []
    quat = Quaternion()
    for i in range(vicon['rots'].shape[-1]):
        rot = vicon['rots'][:,:,i].reshape(3,3)
        q = quat.from_rotm(rot)
        euler_angles = quat.euler_angles()
        vicon_roll.append(euler_angles[0])
        vicon_pitch.append(euler_angles[1])
        vicon_yaw.append(euler_angles[2])
        
        
    x_k_1 = np.zeros((7, 1))		             #x0
    x_k_1[0, 0] = 1
    P_k_1 = np.eye(6)				     # P0

    Q = (np.eye(6)*0.1).astype(float)            # Process noise (6X6)
    R = (np.eye(6)*0.1).astype(float)            # measurement noise (6x6)

    
    roll = []
    pitch = []
    yaw = []
    ox = []
    oy = []
    oz = []
    cov = []
    
    for i in range(T):
 
        X = generate_sigma(x_k_1, P_k_1, Q)       
        #print(X)
        
        Y = transform(X, delta_t[i])
        #print("Y:")
        #print(Y)
        
        x_k_bar = get_mean(Y, X[0:4, 0])
        #print(x_k_bar)
        
        W_dash = get_W(Y, x_k_bar)
        #print(W_dash)
        P_k_bar = (W_dash @ W_dash.T)/12.0

        Z = get_Z(Y)
        #print("Z:")
        #print(Z)

        Z_avg = np.mean(Z, axis=1).reshape(6,1)
        #print(Z_avg)
        #print(yug)
        Z_obs = np.zeros((6,1))
        
        Z_obs[0:3, 0] = np.array([ax[i], ay[i], az[i]])
        Z_obs[3:6, 0] = np.array([gx[i], gy[i], gz[i]])

        m = Z - Z_avg
        P_zz = (m @ m.T)/12.0
        P_vv = (P_zz+R)
        P_xz = (W_dash @ m.T)/12.0
        K = P_xz @ np.linalg.inv(P_vv)

        P_k = P_k_bar - K @ P_vv @ K.T

        v_k = K @ (Z_obs- Z_avg)   # innovation
        
        x_k_bar[4:7,0] += v_k[3:6,0]
        
        q_v_k = Quaternion()
        q_v_k.from_axis_angle(v_k[0:3,0])

        q_x_k_bar = Quaternion(x_k_bar[0,0], x_k_bar[1:4,0])
        q_x_k_bar = q_v_k * q_x_k_bar
        q_x_k_bar.normalize()

        x_k_bar[0:4,0] = q_x_k_bar.q
        
        x_k = x_k_bar

        #print(np.shape(x_k))
        x_k_1 = x_k
        P_k_1 = P_k

        euler_angles = q_x_k_bar.euler_angles()
        roll.append(euler_angles[0])
        pitch.append(euler_angles[1])
        yaw.append(euler_angles[2])
        
        ox.append(x_k_bar[4,0])
        oy.append(x_k_bar[5,0])
        oz.append(x_k_bar[6,0])
        cov.append(np.sqrt(np.diag(P_k_1)))
        print(i)

    # roll, pitch, yaw are numpy arrays of length T

    # roll, pitch, yaw are numpy arrays of length T
    plt.figure('roll')
    plt.plot(vicon_roll, label='vicon roll')
    plt.plot(roll, label='filtered roll')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    
    
    plt.figure('pitch')
    plt.plot(vicon_pitch, label='vicon pitch')
    plt.plot(pitch, label='filtered pitch')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    
    plt.figure('yaw')
    plt.plot(vicon_yaw, label='vicon yaw')
    plt.plot(yaw, label='filtered yaw')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    #plt.show()
    
    plt.figure('ox')
    plt.plot(gx, label='gyroX')
    plt.plot(ox, label='filtered omega x')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.legend()
    
    plt.figure('oy')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.plot(gy, label='gyroY')
    plt.plot(oy, label='filtered omega y')
    plt.legend()
    
    plt.figure('oz')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.plot(gz, label='gyroZ')
    plt.plot(oz, label='filtered omega z')
    plt.legend()
    #plt.show()
    
    k = np.arange(1,T+1)
    cov = np.asarray(cov).T
    plt.figure('q_mc1')
    plt.plot(k, roll)
    #print(roll[:] - cov[0,:])
    plt.fill_between(k, roll[:] - cov[0,:], roll[:] + cov[0,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")

    plt.figure('q_mc2')
    plt.plot(k, pitch)
    plt.fill_between(k, pitch - cov[1,:], pitch + cov[1,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")

    plt.figure('q_mc3')
    plt.plot(k, yaw)
    plt.fill_between(k, yaw - cov[2,:], yaw + cov[2,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    
    plt.figure('o_mc1')
    plt.plot(k, ox)
    plt.fill_between(k, ox - cov[3,:], ox + cov[3,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")

    plt.figure('o_mc2')
    plt.plot(k, oy)
    plt.fill_between(k, oy - cov[4,:], oy + cov[4,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")

    plt.figure('o_mc3')
    plt.plot(k, oz)
    plt.fill_between(k, oz - cov[5,:], oz + cov[5,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.show()
 
    return roll,pitch,yaw
    
    
roll, pitch, yaw = estimate_rot(1)
