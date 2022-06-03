# Robot Localization and Mapping
Course-assignments for the course ESE 650 - Learning in Robotics (Spring 2022) that involve State Estimation (Kalman Filter and its variants) and Mapping (Occupany Grid). The theory behind these concepts is described in these blog posts: [Deriving Kalman Gain](https://yainnoware.blogspot.com/2022/05/kalman-filter-part-1-introduction.html), [Filtering Algorithm](https://yainnoware.blogspot.com/2022/05/kalman-filter-part-2-filtering-algorithm.html), [EKF](https://yainnoware.blogspot.com/2022/05/kalman-filter-part-3-extended-kf.html), [UKF](https://yainnoware.blogspot.com/2022/05/kalman-filter-part-4-unscented-kf.html), [Particle Filter](https://yainnoware.blogspot.com/2022/06/kalman-filter-part-5-particle-filter_1.html), [Mapping](https://yainnoware.blogspot.com/2022/06/mapping-occpany-grid.html).

The problems are described in more detail in [Problems.pdf](https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problems.pdf) and the solutions in [Solutions.pdf](https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Solutions.pdf).


### Problem 1 - Extended Kalman Filter (EKF)
The aim of this problem is to use filtering to estimate an unknown parameter of a non-linear dynamical system. Collected a dataset of observations using the ground truth value of the system parameter and developed the EKF equations that uses this dataset to estimate it. 

<img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%201%20-%20EKF/Figure_2.png" width="400"> 


### Problem 2 - Unscented Kalman Filter (UKF)
Implemented a quaternion-based 7-DOF UKF for tracking the orientation of a drone in three-dimensions using 6-axis IMU data. The Vicon data is used as ground truth for calibration and tuning of the filter. 

<img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%202%20-%20UKF/images/roll.png" width="260"><img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%202%20-%20UKF/images/pitch.png" width="260"><img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%202%20-%20UKF/images/yaw.png" width="260">

### Problem 3 - Simultaneous Localization and Mapping (SLAM) with Particle Filter
Coded a particle filter based SLAM using odometry and lidar data collected on THOR-OP humanoid robot to build a 2D occupancy grid map of indoor environments.

<img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%203%20-%20Particle%20Filter/map1.png" width="400"><img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%203%20-%20Particle%20Filter/map2.png" width="400"> 
<img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%203%20-%20Particle%20Filter/map3.png" width="400"><img src="https://github.com/YugAjmera/Robot-Localization-and-Mapping/blob/main/Problem%203%20-%20Particle%20Filter/map4.png" width="400"> 




