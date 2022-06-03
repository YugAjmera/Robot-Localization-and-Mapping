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
    #vicon = io.loadmat('source/vicon/viconRot'+str(data_num)+'.mat')
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


    x_k_1 = np.zeros((7, 1))		             #x0
    x_k_1[0, 0] = 1
    P_k_1 = np.eye(6)				     # P0

    Q = (np.eye(6)*0.1).astype(float)            # Process noise (6X6)
    R = (np.eye(6)*0.1).astype(float)            # measurement noise (6x6)

    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)
    
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
        roll[i] = euler_angles[0]
        pitch[i] = euler_angles[1]
        yaw[i] = euler_angles[2]
        #print(i)

    # roll, pitch, yaw are numpy arrays of length T

    return roll,pitch,yaw
    
    
roll, pitch, yaw = estimate_rot(1)
