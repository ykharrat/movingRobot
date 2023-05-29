#!/usr/bin/env python
######################################################################
#
# File: particle_filter.py
# Author: Michelle Zhuang and Rey Mendoza
# Written for E28 - Mobile Robotics, fall 2022
#
# Implement a particle filter used to localize a robot equipped 
# with a noisy range sensor that reports ranges to known beacons.

import roslib; roslib.load_manifest('project3')
import rospy
import rospkg
import sys

from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState
from blobfinder2.msg import MultiBlobInfo3D

from transform2d import transform2d_from_ros_transform, distance_2d, Transform2D

import tf
import math
import numpy as np
import matplotlib.pyplot as plt

# control at 100Hz
CONTROL_PERIOD = rospy.Duration(0.01)

# minimum duration of safety stop (s)
STOP_DURATION = rospy.Duration(1.0)

# minimum distance of cones to make gate (m)
MIN_GATE_DIST = 0.75

# maximum distance of cones to make gate (m)
MAX_GATE_DIST = 1.25

# minimum number of pixels to identify cone (px)
MIN_CONE_AREA = 200

# maximum change in linear velocity per 0.01s timestep
MAX_LINEAR_CHANGE = 0.003 

# maximum change in angular velocity per 0.01s timestep
MAX_ANGULAR_CHANGE = 0.05 

######################################################################
# r = true_ranges(x, b)
#
#   x is an n-by-2 array of (x, y) robot locations
#   b is a k-by-2 "map" of known (x, y) beacon locations
#
# returns a k-by-n array of distances r, such that r(i,j) is the
# distance between beacon position i and robot state j.

def true_ranges(x, b):

    if len(x.shape) == 1:
        x = x.reshape((1, 2))

    assert( len(x.shape) == 2 and x.shape[1] == 2 )
    assert( len(b.shape) == 2 and b.shape[1] == 2 )

    # Hellacious numpy broadcasting operations.
    # Vectorize all of the things!
    diff = x.T[None, :, :] - b[:, :, None]
    
    return np.sqrt( (diff ** 2).sum(axis = 1) )


######################################################################
# p = measurement_model(z, x, b, sigma_z)
#
#   z is an array of measured ranges of length k
#   x is an n-by-2 array of (x, y) robot locations
#   b is a k-by-2 "map" of known (x, y) beacon locations
#   sigma_z is the standard deviation of range measurement error
#
# The robot is equipped with a noisy range sensor which reports the
# distance to a number of beacons at locations b. The error on the
# range measurments is Gaussian with standard deviation sigma_z.
#
# This function returns an array p of n un-normalized likelihoods
# proportional to the probability (density) of receiving the
# measurement z in each state given in x.

def measurement_model(z, x, b, sigma_z, sensor_idx=None):

    z = z.flatten()
    assert( b.shape[0] == len(z) and b.shape[1] == 2)
    
    err = true_ranges(x, b) - z[:, None]
    probs = np.exp(-err**2 / (2 * sigma_z**2))

    if sensor_idx is None:
        return probs.prod(axis=0)
    else:
        return probs[sensor_idx,:]

######################################################################
# z = motion_model(x_cur, u_cur, dt)
#
#   x_cur is an n-by-3 array of (x, y, theta) robot locations
#   u_cur is the robot (dx, d_theta) action of length 2
#   dt is time step
def motion_model(x_cur, u_cur, dt):
    x_next = np.zeros_like(x_cur)
    x_next[0] = x_cur[0] + np.cos(x_cur[2])*u_cur[0]*dt
    x_next[1] = x_cur[1] + np.sin(x_cur[2])*u_cur[0]*dt
    x_next[2] = x_cur[2] + u_cur[1]*dt

    return x_next

######################################################################
# Do motion update for particle filter by sampling from the motion
# model. Simply calls the above function.

def particle_motion_update(particles, u, dt):

    particles[:] = motion_modle(particles, u, dt)

######################################################################
# Do measurement update by weighting and resampling.

def particle_measurement_update(z, particles, b, sigma_z):

    # Compute weights by consulting measurement model.
    weights = measurement_model(z, particles, b, sigma_z)

    # Normalize to form a proper probability distribution.
    weights /= weights.sum()

    # Resample particles by using weighted sampling or uniform
    # sampling in box.
    new_particles = np.empty_like(particles)
    
    for i in range(len(particles)):
        j = np.random.choice(len(particles), p=weights)
        new_particles[i] = particles[j]

    particles[:] = new_particles

######################################################################
# Class to implement our particle filter demo. All of the important
# concepts are illustrated above, this really just does plotting and
# bookkeeping.

class Particle_Filter_Control:
    # initialize our particle filter
    
    def __init__(self):
        # set up publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # set up a TransformListener to get odometry information
        self.odom_listener = tf.TransformListener()

    # get current pose from TransformListener
    def get_current_pose(self):

        try:
            ros_xform = self.odom_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:

            return None

        return transform2d_from_ros_transform(ros_xform)
    
    # called by main function below (after init)
    def run(self):

        # timers and callbacks are already set up, so just spin
        rospy.spin()

        # if spin returns we were interrupted by Ctrl+C or shutdown
        rospy.loginfo('goodbye')

    def control_callback(self, timer_event = None):


# main function
if __name__ == '__main__':
    try:
        ctrl = Particle_Filter_Control()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass

# class ParticleFilterDemo:

#     # Construct a demo object
#     def __init__(self, *args):

#         lo = 0.0
#         hi = 100.0

#         plt.rcParams['keymap.xscale'] = []
#         plt.rcParams['keymap.yscale'] = []
        
#         self.bounds = [lo, hi]
#         self.xyrng = np.linspace(lo, hi, 129)
#         self.xymid = 0.5 * (self.xyrng[1:] + self.xyrng[:-1])
#         self.xgrid, self.ygrid = np.meshgrid(self.xymid, self.xymid)

#         self.all_xy = np.vstack( ( self.xgrid.flatten(),
#                                    self.ygrid.flatten() ) ).transpose()

#         print('self.all_xy.shape =', self.all_xy.shape)

#         self.sigmas = [ 1.0, 4.0, 9.0 ]
#         self.sigma_xy_idx = 0
#         self.sigma_z_idx = 1
        
#         self.sigma_xy = self.sigmas[self.sigma_xy_idx]
#         self.sigma_z = self.sigmas[self.sigma_z_idx]

#         self.beacon_locations = np.array([
#             [ 10.0, 10.0 ],
#             [ 90.0, 90.0 ],
#             [ 20.0, 70.0 ],
#             [ 70.0, 30.0 ] ])

#         self.n = 250

#         self.u = np.array([0.0, 0.0])
#         self.step_size = 1.0

#         self.disable_sensing = False
#         self.cheat_option = True
#         self.which_to_vis = None
#         self.did_motion_update = False
        
#         self.uniform_idx = 0
#         self.uniforms = [ 0, 25, 250 ]
#         self.num_uniform = self.uniforms[self.uniform_idx]

#         # Reset the simulation state
#         self.reset()

#         # Rest is to deal with plotting and GUI stuff
#         self.animating = False
#         self.fig = plt.gcf()
#         self.init_plot()
#         self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
#         self.timer = self.fig.canvas.new_timer(interval=5)
#         self.timer.add_callback(self.on_timer)
#         self.timer.start()

#     # Reset the simulation and filter to a well-defined state
#     def reset(self, kidnap=False):

#         self.u = np.array([0.0, 0.0])

#         self.x_true = np.array([ 40, 60 ])
            
#         self.particles = np.tile(self.x_true, (self.n, 1))

#         self.particles = sample_motion(self.particles,
#                                        np.array([0,0]),
#                                        self.sigma_xy)

#         self.z = sample_measurement(self.x_true, self.beacon_locations,
#                                     self.sigma_z, self.cheat_option)

#         self.did_motion_update = False

#         self.status = 'after reset'

#         print('reset simulation back to initial state!')

#     # Perform a single motion or measurement step for first the
#     # simulator, then the particle filter.
#     def step(self): # Note u is ignored if doing motion update

#         if self.disable_sensing or not self.did_motion_update:

#             self.x_true = sample_motion(self.x_true, self.u,
#                                         self.sigma_xy,
#                                         self.cheat_option).flatten()
            
            
#             particle_motion_update(self.particles, self.u,
#                                    self.sigma_xy)


#             self.z = sample_measurement(self.x_true,
#                                         self.beacon_locations,
#                                         self.sigma_z,
#                                         self.cheat_option)

#             self.status = 'after motion'

#             self.did_motion_update = True

#         else:

#             particle_measurement_update(self.z,
#                                         self.particles,
#                                         self.beacon_locations,
#                                         self.sigma_z,
#                                         self.num_uniform,
#                                         self.bounds)

#             self.status = 'after measurement'

#             self.did_motion_update = False

#     def step_u(self, dx, dy):

#         self.u = np.array([dx, dy], dtype=float)*self.step_size
#         self.step()
#         self.update_plot()

#     # Run our demo
#     def run(self):
#         self.print_help_text()
#         plt.show()
            
#     # Print this when program starts or user hits 'h'
#     def print_help_text(self):

#         print('\n'.join([
#             '',
#             '  r ........................ reset',
#             '  k ........................ kidnap robot',
#             '  x ........................ toggle size of sigma for x/y',
#             '  z ........................ toggle size of sigma for z',
#             '  c ........................ toggle cheating',
#             '  m ........................ toggle measurement',
#             '  u ........................ toggle # uniforms',
#             '  1-4 ...................... select which sensor to visualize',
#             '  0 ........................ visualize joint sensor probability',
#             '  arrow keys ............... single step in direction',
#             '  space .................... single step in place',
#             '  enter .................... auto step to right',
#             '  escape ................... quit',
#             '',
#         ]))
            

#     # Called when keyboard pressed in plot
#     def on_key_press(self, event):

#         lkey = event.key.lower()

#         if lkey != 'enter':
#             self.animating = False

#         if lkey == 'escape':
#             print('goodbye')
#             sys.exit(0)
#         elif lkey == 'enter' or lkey == 'return':
#             self.animating = not self.animating
#         elif lkey == 'r':
#             self.reset(kidnap=False)
#             self.update_plot()
#         elif lkey == 'k':
#             self.x_true = np.random.uniform(self.bounds[0],
#                                             self.bounds[1],
#                                             2)
#             self.z = sample_measurement(self.x_true, self.beacon_locations,
#                                         self.sigma_z, self.cheat_option)
#             self.did_motion_update = True
#             self.u = np.array([0.0, 0.0])
#             self.status = 'kidnapped!'
#             self.update_plot()
#         elif lkey == 'x':
#             self.sigma_xy_idx = (self.sigma_xy_idx + 1) % len(self.sigmas)
#             self.sigma_xy = self.sigmas[self.sigma_xy_idx]
#             print('set sigma x/y to', self.sigma_xy)
#             self.update_plot()
#         elif lkey == 'z':
#             self.sigma_z_idx = (self.sigma_z_idx + 1) % len(self.sigmas)
#             self.sigma_z = self.sigmas[self.sigma_z_idx]
#             print('set sigma z to', self.sigma_z)
#             self.update_plot()
#         elif lkey >= '1' and lkey <= '4':
#             self.which_to_vis = ord(lkey) - ord('1')
#             self.update_plot()
#         elif lkey == '0' or (lkey >= '5' and lkey <= '9'):
#             self.which_to_vis = None
#             self.update_plot()
#         elif lkey == 'c':
#             self.cheat_option = not self.cheat_option
#             print('set cheat =', self.cheat_option)
#             self.update_plot()
#         elif lkey == 'left':
#             self.step_u(-1, 0)
#         elif lkey == 'right':
#             self.step_u(1, 0)
#         elif lkey == 'up':
#             self.step_u(0, 1)
#         elif lkey == 'down':
#             self.step_u(0, -1)
#         elif lkey == ' ':
#             self.step_u(0, 0)
#         elif lkey == 'm':
#             print('sensing enabled:', self.disable_sensing)
#             self.disable_sensing = not self.disable_sensing
#             self.update_plot()
#         elif lkey == 'u':
#             self.uniform_idx = (self.uniform_idx + 1) % len(self.uniforms)
#             self.num_uniform = self.uniforms[self.uniform_idx]
#             print('set # uniform to', self.num_uniform)
#             self.update_plot()
#         else:
#             print('ignoring', event.key)


#     # Called periodically
#     def on_timer(self):
        
#         if self.animating:
#             self.step()
#             self.update_plot()

#     # Set up our plot window
#     def init_plot(self):

#         plt.figure(self.fig.number)
#         plt.clf()

#         self.mesh = plt.pcolormesh(self.xyrng, self.xyrng,
#                                    np.zeros_like(self.xgrid),
#                                    vmin=0, vmax=1.0)

#         self.plt_beacons = plt.plot(self.beacon_locations[:,0],
#                                     self.beacon_locations[:,1],
#                                     'wo')

#         self.plt_particles = plt.plot(self.particles[:,0],
#                                       self.particles[:,1],
#                                       'k.', markersize=2)

#         self.plt_marker = plt.plot(self.x_true[0], self.x_true[1], 'mo')

#         plt.axis('equal')
#         plt.axis('off')
        
#         self.title = plt.title('')

#         self.plt_text = plt.text(0.5*self.bounds[1], -0.01*self.bounds[1], 'hi',
#                                  verticalalignment='top',
#                                  horizontalalignment='center')

#         self.update_plot()

#     # Update our plot window to reflect state of simulation and filter
#     def update_plot(self):

#         self.pvis = measurement_model(self.z, self.all_xy,
#                                       self.beacon_locations,
#                                       self.sigma_z, self.which_to_vis)

#         self.pvis = self.pvis.reshape(self.xgrid.shape)

#         self.mesh.set_clim(self.pvis.min(), self.pvis.max())

#         #self.pvis /= self.pvis.max()
        
#         self.plt_marker[0].set_data(self.x_true[0], self.x_true[1])
#         self.plt_particles[0].set_data(self.particles[:,0], self.particles[:,1])

#         self.mesh.set_array(self.pvis.ravel())

#         if self.which_to_vis is None:
#             vis_str = 'joint z'
#         else:
#             vis_str = 'sensor ' + str(self.which_to_vis+1)

#         self.title.set_text(
#             'sense={:}, cheat={:}, sxy={:}, sz={:}, vis={:}, #uni.={:}'.format(
#                 str(not self.disable_sensing),
#                 str(self.cheat_option), self.sigma_xy,
#                 self.sigma_z, vis_str, self.num_uniform))
            
#         self.plt_text.set_text(self.status)
        
#         plt.draw()


        
# ######################################################################
# # Finally, our main function

# if __name__ == '__main__':

#     d = ParticleFilterDemo()
#     d.run()
    

    
