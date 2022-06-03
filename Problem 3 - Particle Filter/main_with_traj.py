
import click, tqdm, random

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.05, Q=np.diag([2e-4,2e-4,1e-4]))
    #slam = slam_t(resolution=0.05, Q=np.diag([2e-4,2e-4,1e-3]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)
    
    #raise NotImplementedError
    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    #### TODO: XXXXXXXXXXX
    t0 = 0
    while(abs(slam.lidar[t0]['t'] < float(slam.joint['t'][0]))):
        t0 += 1
        
    # initialize the occupancy grid using one particle and calling the observation_step
    # function
    #### TODO: XXXXXXXXXXX
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)

    # slam, save data to be plotted later
    #### TODO: XXXXXXXXXXX
    slam.init_particles(n = 100)
    list_x = []
    list_y = []

    list_odom_x = []
    list_odom_y = []

    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        slam.observation_step(t)

        x,y = slam.map.grid_cell_from_xy(slam.current[0], slam.current[1])
        list_x.append(x)
        list_y.append(y)

        xyth = slam.lidar[t]['xyth']
        odom_x,odom_y = slam.map.grid_cell_from_xy(xyth[0], xyth[1])
        list_odom_x.append(odom_x)
        list_odom_y.append(odom_y)

        if(t%200 == 0):
            display_map = 255*np.ones((slam.map.cells.shape[0],slam.map.cells.shape[1],3),dtype=np.uint8)
            ox, oy = np.where(slam.map.cells == 1)  			# occupied idices
            display_map[ox, oy] = [0,0,0]

            ux, uy = np.where(slam.map.num_obs_per_cell == 0)  	# unexplored idices
            display_map[ux, uy] = [150,150,150]

            display_map[list_x,list_y] = [228,70,70]          	# Estimated trajectory


            display_map[list_odom_x,list_odom_y] = [70,70,228]         # Odometry trajectory
            
            plt.imshow(display_map)
            plt.pause(0.01)
    
    
    plt.show()
    return 0

@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main()
