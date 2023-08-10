#! /usr/bin/env python3
import math
import csv
import ctypes
import numpy as np
import os

import sys
import transformations as T
import yaml
import argparse
from timeit import default_timer as timer

from math_utils import get_quaternion_from_euler, euler_from_quaternion, slerp, unpack_pose_xyz_euler

parser = argparse.ArgumentParser(
                    prog='traj_tracing',
                    )
parser.add_argument('--no_ros',
                    action='store_true')

parser.add_argument('--init',type=str)

args = parser.parse_args()

if not args.no_ros:
    import rospkg
    import rospy
    from geometry_msgs.msg import Pose, Twist, Vector3
    from sensor_msgs.msg import JointState
    from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals, IKUpdateWeight
    from relaxed_ik_ros1.srv import IKPoseRequest,  IKPose
else:
    from math_utils import Pose7d as Pose
    # make python find the package
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/../')
    from relaxed_ik_rust_demo import RelaxedIKDemo as ik_solver

from robot import Robot
from robot_utils import movo_jointangles_fik2rik, movo_jointangles_rik2fik

if args.no_ros:
    path_to_src = os.path.dirname(os.path.abspath(__file__)) + '/../relaxed_ik_core'
else:
    path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'


class TraceALine:
    def __init__(self, initial_poses=None):
        try:
            tolerances = rospy.get_param('~tolerances')
        except:
            print("No tolerances are given, using zero tolerances")
            tolerances = [0, 0, 0, 0, 0, 0]

        
        try:
            self.use_topic_not_service = True#rospy.get_param('~use_topic_not_service')
            print("use_topic_not_service")
        except:
            self.use_topic_not_service = False

        try: 
            self.loop = rospy.get_param('~loop')
        except:
            self.loop = False
        
        self.tolerances = []
        self.time_between = 2.0   # time between two keypoints
        self.num_per_goal = 50    # number of intermediate points between two keypoints
        self.start_from_init_pose = True
        traj_offset = np.array([0,0,0.0,0,0,0])

        assert len(tolerances) % 6 == 0, "The number of tolerances should be a multiple of 6"
        for i in range(int(len(tolerances) / 6)):
            if not args.no_ros:  
                self.tolerances.append(Twist(   Vector3(tolerances[i*6],    tolerances[i*6+1], tolerances[i*6+2]), 
                                                Vector3(tolerances[i*6+3],  tolerances[i*6+4], tolerances[i*6+5])))
            else:
                self.tolerances.append([tolerances[i*6],    tolerances[i*6+1], tolerances[i*6+2], 
                                        tolerances[i*6+3],  tolerances[i*6+4], tolerances[i*6+5]])

        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        # setting_file_path = rospy.get_param('setting_file_path')
        setting_file_path = ''
        if setting_file_path == '':
            setting_file_path = deault_setting_file_path

        os.chdir(path_to_src)
        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
        
        urdf_file = open(path_to_src + '/configs/urdfs/' + settings["urdf"], 'r')
        urdf_string = urdf_file.read()
        if not args.no_ros:
            rospy.set_param('robot_description', urdf_string)

        self.robot = Robot(setting_file_path, use_ros=not args.no_ros, path_to_src=path_to_src)
        self.chains_def = settings['chains_def']
        
        if args.no_ros:
            self.ik_solver = ik_solver(path_to_src)
            
        
        if initial_poses is not None:
            if args.no_ros:
                min_start_loss = float('inf')
                best_starting_config = None
                for initial_pose in initial_poses:
                    conf = list(movo_jointangles_fik2rik(initial_pose))
                    self.ik_solver.reset(conf)
                    loss = self.ik_solver.query_loss(conf)
                    # print(loss)
                    if loss < min_start_loss:
                        best_starting_config = conf
                        min_start_loss = loss
                
                starting_config = best_starting_config
                print(f"Min start loss = {min_start_loss}")
            else:
                raise NotImplementedError()
        else:
            starting_config = settings['starting_config']
        starting_config_translated = self.translate_config(starting_config, self.chains_def)
        self.starting_ee_poses = self.robot.fk(starting_config_translated)
        print(starting_config)
        print(self.starting_ee_poses[0])
        print(self.starting_ee_poses[1])
        
        trajs = []
        for traj_file in settings["traj_files"]:
            trajs.append(np.load(path_to_src + '/traj_files/' + traj_file) + traj_offset)

        # print(trajs[0].shape, trajs[1].shape)
        
        
        ### set initial positions
        # self.init_position = [[0.8,-0.5,0.8],[0.8,0.5,0.8]]
        # self.init_orientation = [[0,0,0],[0,0,0]]
        p0, p1 = unpack_pose_xyz_euler(self.starting_ee_poses[0]), unpack_pose_xyz_euler(self.starting_ee_poses[1])
        self.init_position    = [p0[0], p1[0]]
        self.init_orientation = [p0[1], p1[1]] 
        
        trajs_with_init = []
        traj_lengths = []
        
        if self.start_from_init_pose:
            for i, traj in enumerate(trajs):
                init_pos = np.array(self.init_position[i] + self.init_orientation[i])
                trajs_with_init.append(np.vstack([init_pos, init_pos, traj[0], traj]))
                # trajs_with_init.append(np.vstack([init_pos, init_pos]))
                traj_lengths.append(len(trajs_with_init[i]))
        else:
            trajs_with_init = trajs
            for i, traj in enumerate(trajs):
                traj_lengths.append(len(trajs[i]))
        # print(trajs_with_init)
        
        # fill trajectory with initial position if provided trajs are less than num of arms
        if len(trajs_with_init) < self.robot.num_chain:
            shape0 = trajs_with_init[0].shape
            for i in range(len(trajs_with_init), self.robot.num_chain):
                trajs_with_init.append(np.tile(np.array(self.init_position[i] + self.init_orientation[i]), shape0))
                traj_lengths.append(len(trajs_with_init[i]))
        
        assert(all(l == traj_lengths[0] for l in traj_lengths))
        
        self.num_keypoints = traj_lengths[0]
        self.trajectory = self.generate_trajectory(trajs_with_init, self.num_per_goal)
        self.weight_updates = self.generate_weight_updates(self.num_keypoints, self.num_per_goal)
        self.trajectory_index = 0
        print(len(self.trajectory),len(self.weight_updates))
        assert(len(self.trajectory) == len(self.weight_updates))
        
        if not args.no_ros:
            # ROS is present
            if self.use_topic_not_service:
                self.ee_pose_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
                self.ik_weight_pub = rospy.Publisher('relaxed_ik/ik_update_weight', IKUpdateWeight, queue_size=128)
                self.ik_reset_pub = rospy.Publisher('relaxed_ik/reset', JointState, queue_size=5)
                self.query_loss_pub = rospy.Publisher('relaxed_ik/query_loss', JointState, queue_size=5)
            else:
                rospy.wait_for_service('relaxed_ik/solve_pose')
                self.ik_pose_service = rospy.ServiceProxy('relaxed_ik/solve_pose', IKPose)
            
            loss = self.query_loss(starting_config)
            print("Loss:", loss)
            self.reset(starting_config)
        
            count_down_rate = rospy.Rate(1)
            count_down = 3
            while not rospy.is_shutdown():
                print("Start line tracing in {} seconds".format(count_down))
                count_down -= 1
                if count_down == 0:
                    break
                count_down_rate.sleep()

            self.timer = rospy.Timer(rospy.Duration(self.time_between / self.num_per_goal), self.timer_callback)
            self.ik_solver = None
            
        else:
            # No ROS, initialize the IK in this process:
            # print(self.ik_solver.weight_names)
            # loss = self.ik_solver.query_loss(starting_config)
            # print(f"Initial Loss: {loss}")
            pass
            # print(self.trajectory)
            
    def generate_trajectory(self, trajs, num_per_goal):
        
        assert len(trajs) == self.robot.num_chain
        trajectory = []
        
        
        for i in range(len(trajs[0]) - 1):
            for t in np.linspace(0, 1, num_per_goal):
                poses = self.copy_poses(self.starting_ee_poses)
                for k in range(self.robot.num_chain):
                    traj = trajs[k]
                    # linear interpolation
                    position_goal = (1 - t) *np.array(traj[i][:3]) + t * np.array(traj[i+1][:3]) 
                    orientation_goal = slerp(np.array(get_quaternion_from_euler(traj[i][3], traj[i][4], traj[i][5])),
                                             np.array(get_quaternion_from_euler(traj[i+1][3], traj[i+1][4], traj[i+1][5])),
                                             t)
                    ( poses[k].position.x, 
                      poses[k].position.y, 
                      poses[k].position.z )    = tuple(position_goal)
                    
                    ( poses[k].orientation.x, 
                      poses[k].orientation.y,
                      poses[k].orientation.z,
                      poses[k].orientation.w ) = tuple(orientation_goal)

                trajectory.append(poses)
            
        return trajectory

    def generate_weight_updates(self, num_keypoints, num_per_goal):
        """
        Update weight after the first IK (for more smoothness)
        """
        weight_updates = []
        
        for i in range(num_keypoints - 1):
            num_empty_updates = num_per_goal
            if self.start_from_init_pose and i == 0:
                weight_updates.append({
                    'eepos' : 50.0,
                    'eequat' : 1.0,
                    'minvel'  : 0.5,
                    'minacc'  : 0.3,
                    'minjerk' : 0.1,
                    'selfcollision' : 0.01,
                    'selfcollision_ee' : 0.05,
                    'envcollision': 0.5,
                    'maxmanip' : 3.0,
                })
                num_empty_updates -= 1
            elif self.start_from_init_pose and i == 1:
                weight_updates.append({
                    'eequat'  : 3.0,
                    'minvel'  : 0.7,
                    'minacc'  : 0.5,
                    'minjerk' : 0.3,
                    'selfcollision_ee' : 0.5,
                    'envcollision': 10.0,
                    'jointlimit' : 3.0,
                })
                num_empty_updates -= 1
            for _ in range(num_empty_updates):
                weight_updates.append({})
        
        return weight_updates
                
    
    def copy_poses(self, input_poses):
        output_poses = []
        for i in range(len(input_poses)):
            output_poses.append(self.copy_pose(input_poses[i]))
        return output_poses
    
    def copy_pose(self, input_pose):
        output_pose = Pose()
        output_pose.position.x = input_pose.position.x
        output_pose.position.y = input_pose.position.y
        output_pose.position.z = input_pose.position.z
        output_pose.orientation.x = input_pose.orientation.x
        output_pose.orientation.y = input_pose.orientation.y
        output_pose.orientation.z = input_pose.orientation.z
        output_pose.orientation.w = input_pose.orientation.w
        return output_pose
    
    def reset(self, config):
        # reset
        print("RESET:", config)
        js_msg = JointState()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = self.robot.articulated_joint_names
        js_msg.position = config
        self.ik_reset_pub.publish(js_msg)
    
    def query_loss(self, config):
        js_msg = JointState()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = self.robot.articulated_joint_names
        js_msg.position = config
        self.ik_reset_pub.publish(js_msg)
    
    def timer_callback(self, event):
        if self.trajectory_index >= len(self.trajectory):
            if self.loop:
                print("Trajectory finished, looping")
                self.trajectory_index = 0
            else:
                rospy.signal_shutdown("Trajectory finished")
            return

        if self.use_topic_not_service:
            ee_pose_goals = EEPoseGoals()
            for i in range(self.robot.num_chain):
                ee_pose_goals.ee_poses.append(self.trajectory[self.trajectory_index][i])
                if i < len(self.tolerances):
                    ee_pose_goals.tolerances.append(self.tolerances[i])
                else:
                    ee_pose_goals.tolerances.append(self.tolerances[0])
            self.ee_pose_pub.publish(ee_pose_goals)
            
            weight_update = self.weight_updates[self.trajectory_index]
            for k, v in weight_update.items():
                msg = IKUpdateWeight()
                msg.weight_name = k
                msg.value = v
                self.ik_weight_pub.publish(msg)
        else:
            req = IKPoseRequest()
            for i in range(self.robot.num_chain):
                req.ee_poses.append(self.trajectory[self.trajectory_index][i])
                if i < len(self.tolerances):
                    req.tolerances.append(self.tolerances[i])
                else:
                    req.tolerances.append(self.tolerances[0])
            
            ik_solutions = self.ik_pose_service(req)

        self.trajectory_index += 1
    
    def get_ik_list_from_traj(self) -> list:
        """
        Generate list of IK solutions from trajectory without the need of ROS.
        """
        assert self.ik_solver is not None, "IK Solver not initialized."
        assert args.no_ros, "This method is for no-ROS mode"
        ik_solutions = []
        for j in range(len(self.trajectory)):
            positions = []
            orientations = []
            tolerances = []
            for i in range(self.robot.num_chain):
                positions.extend(self.trajectory[j][i].position.tolist())
                orientations.extend(self.trajectory[j][i].orientation.tolist())
                if i < len(self.tolerances):
                    tolerances.extend(self.tolerances[i])
                else:
                    tolerances.extend(self.tolerances[0])
            
            self.ik_solver.update_objective_weights(self.weight_updates[j])
            
            # print(positions, orientations, tolerances)
            ik_solution = self.ik_solver.solve_pose_goals(positions, orientations, tolerances)
            ik_solutions.append(ik_solution)
            loss = self.ik_solver.query_loss(ik_solution)
            print(f"Loss: {loss}")
            print(j)
            
        return ik_solutions
    
    def translate_config(self, joint_angles, chains_def):
        """
        Handle cases where there are duplicate articulated joints in different chains
        """
        ja_out = []
        for chain in chains_def:
            for joint_idx in chain:
                ja_out.append(joint_angles[joint_idx])
                
        return ja_out
    
if __name__ == '__main__':
    fpath = os.path.dirname(os.path.abspath(__file__))
    if not args.no_ros:
        rospy.init_node('LineTracing')
    trace_a_line = TraceALine(np.load(args.init))
    
    if not args.no_ros:
        rospy.spin()
        
    if args.no_ros:
        
        ik_list = trace_a_line.get_ik_list_from_traj()
        ik_list = [movo_jointangles_rik2fik(x) for x in ik_list]
        
        ik_arr = np.array(ik_list)
        print(ik_arr.shape)
        np.save(fpath + '/ik_seq.npy', ik_arr)
        
