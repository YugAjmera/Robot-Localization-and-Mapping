import numpy as np
import matplotlib.pyplot as plt

# Given variables
R = 1
Q = 0.5
a = -1

# Collect dataset - Part a
dataset = [] 
np.random.seed(1)
x_0 = np.random.normal(1,2)
x_k = a * x_0 + np.random.normal(0,1)
x_1 = x_k
for i in range(100):
	x_next = a * x_k + np.random.normal(0,1)
	y_k = np.sqrt(x_k**2 + 1) + np.random.normal(0,0.5)
	dataset.append(y_k)
	x_k = x_next
	
#print(dataset)


# EKF for estimating a - Part b
prev_cov = np.diag([2, 1])	 											
beta = 0.1
mean_a = -10
mean_x = 1

true_value = []
estimated_value_1 = []
estimated_value_2 = []
mu = []

for i in range(100):
	
	# Propagation
	A = np.array([[mean_a, mean_x],[0,1]])
	
	mean = np.array([[mean_a * mean_x],[mean_a]])
	cov = A @ prev_cov @  A.T + np.diag([R, beta])
	
	mean_x = mean[0,0]
	mean_a = mean[1,0]
	
	# Incorporating observation
	C = np.zeros((1,2))
	C[0,0] = mean_x/np.sqrt(mean_x**2 + 1)
	
	K = cov @ C.T @ np.linalg.inv(C @ cov @ C.T + Q)
	
	updated_mean = mean + K * (dataset[i] - np.sqrt(mean_x ** 2 + 1))
	
	updated_cov = (np.eye(2,2) - K @ C) @ cov
	
	prev_cov = updated_cov
	mean_x = updated_mean[0,0]
	mean_a = updated_mean[1,0]
	
	#Z[0,0] = np.random.normal(mean_x, updated_cov[0,0])
	#Z[1,0] = np.random.normal(mean_a, updated_cov[1,1])
	
	true_value.append(-1)
	estimated_value_1.append(mean_a - np.sqrt(updated_cov[1,1]))
	estimated_value_2.append(mean_a + np.sqrt(updated_cov[1,1]))
	mu.append(mean_a)

k = np.arange(1,101)	
plt.plot(k, true_value, label="true value")
plt.plot(k, mu, label="mean value")
plt.xlabel("Time steps k")
plt.ylabel("System parameter a")
plt.legend()
plt.fill_between(k, estimated_value_1, estimated_value_2, alpha=0.2, color='red')
plt.show()

