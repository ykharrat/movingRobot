#!/usr/bin/env python
# ----------------------------------------------------- #
# Final Project
# ----------------------------------------------------- #
# Team Robot: Zora
# Group Member Name and ID: Michelle Zhuang
# Group Member Name and ID: Rey Mendoza

######################################################################
# Import Libraries


import roslib; roslib.load_manifest('project2')
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
from matplotlib import pyplot as plt

######################################################################
# Global parameters

# Image size
image_size = {'width': 640, 'height': 480}

# control at 100Hz
CONTROL_PERIOD = rospy.Duration(0.01)

# minimum duration of safety stop (s)
STOP_DURATION = rospy.Duration(1.0)

# minimum number of pixels to identify cone (px)
MIN_CONE_AREA = 200

x1, y1 = [], []
x2, y2 = [], []


######################################################################
# Helper functions


def clamp(x, lo, hi):
    #Returns lo, if x < lo; hi, if x > hi, or x otherwise"""
    return max(lo, min(x, hi))

def filter_vel(prev_vel, desired_vel, max_change):
    #Update prev_val towards desired_vel at a maximum rate of max_change

    #This should return desired_vel if absolute difference from prev_vel is
    #less than max_change, otherwise returns prev_vel +/- max_change,
    #whichever is closer to desired_vel.

    return clamp(desired_vel,prev_vel-max_change,prev_vel+max_change)
    #return desired_vel


def angle_from_vec(v):
  #Construct the angle between the vector v and the x-axis."""
  return np.arctan2(v[1], v[0])


def offset_vec(v, r):
    #Offset a given vector v by a fixed distance r."""
    return v * (1 + r / np.linalg.norm(v))

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
def motion_model(x_cur, u_cur, dt,sigma_xy,theta):
    x_next = np.zeros_like(x_cur)
    #x_next[:,0] = x_cur[:,0] + np.cos(x_cur[:,2])*(u_cur[0]*dt)
    #x_next[:,1] = x_cur[:,1] + np.sin(x_cur[:,2])*(u_cur[0]*dt)
    #x_next[:,2] = x_cur[:,2] + u_cur[1]*dt
    x_next[:,0] = x_cur[:,0] + np.cos(theta)*(u_cur[0]*dt)
    x_next[:,1] = x_cur[:,1] + np.sin(theta)*(u_cur[0]*dt)
    noise = np.random.normal(scale=sigma_xy, size=x_next.shape)
    return x_next + noise

######################################################################
# Do motion update for particle filter by sampling from the motion
# model. Simply calls the above function.

def particle_motion_update(particles, u, dt,sigma_xy,theta):

    particles[:] = motion_model(particles, u, dt, sigma_xy,theta)

    return particles

######################################################################
# Do measurement update by weighting and resampling.

def particle_measurement_update(z, particles, b, sigma_z):
    #rospy.loginfo(particles)
    # Compute weights by consulting measurement model.
    weights = measurement_model(z, particles, b, sigma_z)

    # Normalize to form a proper probability distribution.
    weights /= weights.sum()

    # Resample particles by using weighted sampling or uniform
    # sampling in box.
    new_particles = np.empty_like(particles)
    for i in range(particles.shape[0]):
        j = np.random.choice(particles.shape[0], p=weights)
        new_particles[i,:] = particles[j,:]

    particles = new_particles
    return particles


######################################################################
# Controller class


class Controller:
    #Class to handle our simple controller"""

    # initialize our controller
    def __init__(self):

        # initialize our ROS node
        rospy.init_node('starter')

        # set up publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # set up a TransformListener to get odometry information
        self.odom_listener = tf.TransformListener()

        # record whether we should stop for safety
        self.should_stop = 0
        self.time_of_stop = rospy.get_rostime() - STOP_DURATION

        # initialize a dictionary of two empty lists of cones. each
        # list should hold world-frame locations of cone in XY
        self.cone_locations = []

        # these bool values will be set to True when we get a new cone message
        self.cones_updated = False

        # start pose
        self.start_pose = None

        # Start at 0 velocity
        self.prev_cmd_vel = Twist()

        # Motion smoothing
        # maximum change in linear velocity per 0.01s timestep
        self.MAX_LINEAR_CHANGE = 0.01

        # maximum change in angular velocity per 0.01s timestep
        self.MAX_ANGULAR_CHANGE = 0.3

        self.k_theta_turn = 0.35
        self.b_theta = 0.5
        self.kx = 0.4
        self.bx = 0.25
        self.k_theta_drive = 1.5

        self.correct_angle = False
        self.correct_position = False
        # set up our trivial 'state machine' controller
        rospy.Timer(CONTROL_PERIOD,
                    self.control_callback)

        # set up subscriber for sensor state for bumpers/cliffs
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        # set up subscriber for yellow cones, green cones
        rospy.Subscriber('/blobfinder2/blobs3d',
                         MultiBlobInfo3D, self.blobs3d_callback)





        self.z = None
        self.u = [0,0]
        self.sigma_z_cones = 0.025
        self.sigma_z_odometry = 0.05
        self.sigma_xy = 0.001
        self.n = 30
        self.particles = np.zeros((self.n,2))
        self.dt = 0
        self.beacon_locations = np.array([[1,0],[1,-1],[0,-1]])
        self.points = [[0.5,-0.5],[1.5,-0.5],[3,0],[3.5,-1],[1,-0.5]]
        self.point_index=0

        self.last_time = None
        self.cur_time = None


    #################################################################
    # Sensor callback functions

    def sensor_callback(self, msg):
     #Called when sensor msgs received

     #Just copy sensor readings to class member variables"""

        if msg.bumper & SensorState.BUMPER_LEFT:
            rospy.loginfo('***LEFT BUMPER***')
        if msg.bumper & SensorState.BUMPER_CENTRE:
            rospy.loginfo('***MIDDLE BUMPER***')
        if msg.bumper & SensorState.BUMPER_RIGHT:
            rospy.loginfo('***RIGHT BUMPER***')
        if msg.cliff:
            rospy.loginfo('***CLIFF***')

        if msg.bumper or msg.cliff:
            self.should_stop = True
            self.time_of_stop = rospy.get_rostime()
        else:
            self.should_stop = False


    def blobs3d_callback(self, msg):
        #Called when a blob message comes in

        #Calls other methods to process blobs"""

        num = len(msg.color_blobs)
        for i in range(num):
            color_blob = msg.color_blobs[i]
            color = color_blob.color.data
            if color == 'red_tape':
                # Don't need to look at red tape for this lab.
                pass
            elif color == 'yellow_cone':
                self.cone_callback(color_blob, 'yellow')
            elif color == 'green_cone':
                self.cone_callback(color_blob, 'green')

    def cone_callback(self, bloblist, color):
        #Called when a cone related blob message comes in

        #Sets cones_updated to true if cones detected. Updates cone_locations
        #by appopriate color"""

        T_world_from_robot = self.get_current_pose()
        if T_world_from_robot is None:
            rospy.logwarn('no xform yet in blobs3d_callback')
            return

        blob_locations = []
        z = []

        for blob3d in bloblist.blobs:
            if blob3d.have_pos and blob3d.blob.area > MIN_CONE_AREA and blob3d.blob.cy > 240:
                blob_in_robot_frame = np.array([blob3d.position.z, -blob3d.position.x])
                blob_dir = blob_in_robot_frame / np.linalg.norm(blob_in_robot_frame)
                blob_in_robot_frame += blob_dir * 0.04 # offset radius
                blob_locations.append( T_world_from_robot * blob_in_robot_frame )
                z.append(np.sqrt(blob_in_robot_frame[0]**2+blob_in_robot_frame[1]**2))
        self.z = np.array(z)
        self.cone_locations = blob_locations
        self.cones_updated = True

    # get current pose from TransformListener
    def get_current_pose(self):

        try:

            ros_xform = self.odom_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:

            return None

        return transform2d_from_ros_transform(ros_xform)

    #################################################################
    # Control Function
    # Turn towards
    def turn_towards(self, point):
        cmd_vel = Twist()
        tx,ty = point
        beta = angle_from_vec(point)
        angularv=self.k_theta_turn*beta+self.b_theta*np.sign(beta)
        cmd_vel.angular.z = angularv
        if np.abs(beta)>=0.01745:
            arrived = False
        else:
            arrived = True
        # Make the robot turn towards the specified point
        return cmd_vel, arrived

    # Drive towards
    def drive_towards(self, point):
        cmd_vel = Twist()
        tx,ty = point
        beta = angle_from_vec(point)
        cmd_vel.linear.x = self.kx*tx+self.bx
        if np.abs(tx)>= 0.1:
            cmd_vel.angular.z = self.k_theta_drive*beta
            arrived=False
        else:
            arrived = True
        # Make the robot turn towards the specified point

        return cmd_vel, arrived

    # called periodically to do top-level coordination of behaviors
    def control_callback(self, timer_event=None):

        # initialize vel to 0, 0
        cmd_vel = Twist()

        time_since_stop = rospy.get_rostime() - self.time_of_stop

        cur_pose = self.get_current_pose()
    
        if cur_pose is None:
            return
        
        if self.start_pose is None:
            self.start_pose = cur_pose


        if self.should_stop or time_since_stop < STOP_DURATION:

            rospy.loginfo('stopped')

        else: # not stopped for safety

            ############################################################################################
            # FINAL PROJECT TODO: STILL NEED TO CREATE ALL THE SELF.BLANK THINGS FOR ANY OF THIS TO WORK
            ############################################################################################
            

            odom_loc = self.start_pose.inverse().compose_with(cur_pose)
            rospy.loginfo(odom_loc)
            x2.append(odom_loc.x)
            y2.append(odom_loc.y)
            odom_beacon = np.array([[odom_loc.x,odom_loc.y]])
            self.particles = particle_measurement_update(np.array([[0]]),
                                            self.particles,
                                            odom_beacon,
                                            self.sigma_z_odometry)
            if self.cones_updated:
                # Currently cone_locations is the location of the cones in world coordinates
                # We need to match a spotted cone to the beacon it corresponds to, and calculate the distance to the cone
                # First initialize our measurements
                beacons = np.zeros((len(self.z),2))
                assert( len(self.z) == len(self.cone_locations) )
                #for cone in self.cone_locations:]
                cones = self.cone_locations
                z = self.z
                for j in range(len(cones)):
                    cone_location = self.start_pose.transform_inv(cones[j])
                    distance = 1000000
                    for i in range(self.beacon_locations.shape[0]):
                        #calculate distance between cone and beacon
                        error = np.linalg.norm(cone_location[0:2]-self.beacon_locations[i,:])
                        if error < distance:
                            index = i
                            distance = error
                    beacons[j,:] = self.beacon_locations[index,:]
                particle_measurement_update(z,
                                            self.particles[:,:2],
                                            beacons,
                                            self.sigma_z_cones)
                                            
                self.cones_updates = False
            average_particle = np.average(self.particles,axis=0)
            rospy.loginfo("average particle:")
            x1.append(average_particle[0])
            y1.append(average_particle[1])
            rospy.loginfo(average_particle)


            world_point_test = self.points[self.point_index]
            point = cur_pose.transform_inv(self.start_pose.transform_fwd(world_point_test))
            #rospy.loginfo(point)
            
            if not self.correct_angle:
                cmd_vel, self.correct_angle = self.turn_towards(point)
            if self.correct_angle and not self.correct_position:
                cmd_vel, self.correct_position = self.drive_towards(point)
            if self.correct_angle and self.correct_position:
                if self.point_index < len(self.points)-1:
                    self.point_index += 1
                    self.correct_angle = False
                    self.correct_position = False
                else:
                    rospy.loginfo('Success!')
                    
            # now filter large changes in velocity before commanding
            # robot - note we don't filter when stopped
            cmd_vel.linear.x = filter_vel(self.prev_cmd_vel.linear.x,
                                          cmd_vel.linear.x,
                                          self.MAX_LINEAR_CHANGE)

            cmd_vel.angular.z = filter_vel(self.prev_cmd_vel.angular.z,
                                           cmd_vel.angular.z,
                                           self.MAX_ANGULAR_CHANGE)

        if self.last_time == None:
                self.last_time = rospy.get_time()
        self.cur_time = rospy.get_time()
        self.dt = self.cur_time-self.last_time
        theta = (self.start_pose.inverse().compose_with(cur_pose)).theta
        self.particles = particle_motion_update(self.particles, self.u,self.dt,
                                    self.sigma_xy,theta)
        self.last_time = self.cur_time
        self.cmd_vel_pub.publish(cmd_vel)
        self.prev_cmd_vel = cmd_vel
        self.u[0] = cmd_vel.linear.x
        self.u[1] = cmd_vel.angular.z


    #################################################################
    # ROS related functions


    def run(self):
        #Called by main function below (after init)"""

        # timers and callbacks are already set up, so just spin
        rospy.spin()

        # if spin returns we were interrupted by Ctrl+C or shutdown
        rospy.loginfo('goodbye')
        plt.plot(x1,y1, label = 'Particle Filter')
        plt.plot(x2,y2, label = 'Odometetry')
        plt.xlabel("Position (x)")
        plt.ylabel("Position (y)")
        plt.legend()
        plt.axis('equal')
        plt.show()

######################################################################
# Main function

if __name__ == '__main__':
    try:
        ctrl = Controller()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass