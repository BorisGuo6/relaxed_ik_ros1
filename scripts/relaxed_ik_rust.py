#! /usr/bin/env python3

import ctypes
import numpy as np
import os, time
import rospkg
import rospy
import sys
import transformations as T
import yaml


from relaxed_ik_ros1.srv import IKPose, IKPoseResponse
from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals, IKUpdateWeight, ResetJointAngles
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState 
from urdf_parser_py.urdf import URDF
from kdl_parser import kdl_tree_from_urdf_model
import PyKDL as kdl
from robot import Robot
from visualization_msgs.msg import InteractiveMarkerFeedback, InteractiveMarkerUpdate

path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
sys.path.insert(1, path_to_src + '/wrappers')
from python_wrapper import RelaxedIKRust, lib


class RelaxedIK:
    def __init__(self):
        rospy.sleep(1)

        default_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = ""
        try: 
            setting_file_path = rospy.get_param('~setting_file_path')
        except:
            pass

        if setting_file_path == "":
            print("Rviz viewer: no setting file path is given, using the default setting file -- {}".format(default_setting_file_path))
            setting_file_path = default_setting_file_path

        try: 
            self.use_visualization = rospy.get_param('~use_visualization')
        except:
            self.use_visualization = False

        os.chdir(path_to_src )

        # Load the infomation
        
        print("setting_file_path: ", setting_file_path)
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
       
        urdf_file = open(path_to_src + '/configs/urdfs/' + settings["urdf"], 'r')
        urdf_string = urdf_file.read()
        rospy.set_param('robot_description', urdf_string)

        self.relaxed_ik = RelaxedIKRust(setting_file_path)

        # Services
        self.ik_pose_service = rospy.Service('relaxed_ik/solve_pose', IKPose, self.handle_ik_pose)

        # Publishers
        self.angles_pub = rospy.Publisher('relaxed_ik/joint_angle_solutions', JointState, queue_size=1)
        if self.use_visualization:
            self.vis_ee_pub = rospy.Publisher('relaxed_ik/vis_ee_poses', EEPoseGoals, queue_size=1)

        self.robot = Robot(setting_file_path)

        self.js_msg = JointState()
        self.js_msg.name = self.robot.articulated_joint_names
        self.js_msg.position = []

        if 'starting_config' not in settings:
            settings['starting_config'] = [0.0] * len(self.js_msg.name)
        else:
            assert len(settings['starting_config']) == len(self.js_msg.name), \
                    "Starting config length does not match the number of joints"
            for i in range(len(self.js_msg.name)):
                self.js_msg.position.append( settings['starting_config'][i] )
        
        self.weight_names  = self.relaxed_ik.get_objective_weight_names()
        self.weight_priors = self.relaxed_ik.get_objective_weight_priors()
        
        # Subscribers
        rospy.Subscriber('/relaxed_ik/ee_pose_goals', EEPoseGoals, self.pose_goals_cb)
        rospy.Subscriber('/relaxed_ik/ee_vel_goals', EEVelGoals, self.pose_vels_cb)
        rospy.Subscriber('/relaxed_ik/reset_ja', ResetJointAngles, self.reset_cb)
        rospy.Subscriber('/relaxed_ik/ik_update_weight', IKUpdateWeight, self.ik_update_weight_cb)
        
        rospy.Subscriber('/simple_marker/feedback', InteractiveMarkerFeedback, self.marker_feedback_cb)
        rospy.Subscriber('/simple_marker/update', InteractiveMarkerUpdate, self.marker_update_cb)
        
        print("\nSolver RelaxedIK initialized!\n")

    def get_ee_pose(self):
        ee_poses = self.relaxed_ik.get_ee_positions()
        ee_poses = np.array(ee_poses)
        ee_poses = ee_poses.reshape((len(ee_poses)//6, 6))
        ee_poses = ee_poses.tolist()
        return ee_poses

    def handle_ik_pose(self, req):
        positions = []
        orientations = []
        tolerances = []
        for i in range(len(req.ee_poses)):
            positions.append(req.ee_poses[i].position.x)
            positions.append(req.ee_poses[i].position.y)
            positions.append(req.ee_poses[i].position.z)
            orientations.append(req.ee_poses[i].orientation.x)
            orientations.append(req.ee_poses[i].orientation.y)
            orientations.append(req.ee_poses[i].orientation.z)
            orientations.append(req.ee_poses[i].orientation.w)
            if i < len(req.tolerances):
                tolerances.append(req.tolerances[i].linear.x)
                tolerances.append(req.tolerances[i].linear.y)
                tolerances.append(req.tolerances[i].linear.z)
                tolerances.append(req.tolerances[i].angular.x)
                tolerances.append(req.tolerances[i].angular.y)
                tolerances.append(req.tolerances[i].angular.z)
            else:
                for j in range(6):
                    tolerances.append(0.0)

        if self.use_visualization:
            vis_msg = EEPoseGoals()
            vis_msg.ee_poses = req.ee_poses
            vis_msg.tolerances = req.tolerances
            self.vis_ee_pub.publish(vis_msg)

        ik_solution = self.relaxed_ik.solve_position(positions, orientations, tolerances)
        self.js_msg.header.stamp = rospy.Time.now()
        self.js_msg.position = ik_solution
        self.angles_pub.publish(self.js_msg)
        res = IKPoseResponse()
        res.joint_state = ik_solution

        return res

    def reset_cb(self, msg):
        print("got reset msg!")
        n = len(msg.joint_angles)
        x = (ctypes.c_double * n)()
        for i in range(n):
            x[i] = msg.joint_angles[i]
        self.relaxed_ik.reset(x)
        


    def pose_goals_cb(self, msg):
        positions = []
        orientations = []
        tolerances = []
        for i in range(len(msg.ee_poses)):
            positions.append(msg.ee_poses[i].position.x)
            positions.append(msg.ee_poses[i].position.y)
            positions.append(msg.ee_poses[i].position.z)
            orientations.append(msg.ee_poses[i].orientation.x)
            orientations.append(msg.ee_poses[i].orientation.y)
            orientations.append(msg.ee_poses[i].orientation.z)
            orientations.append(msg.ee_poses[i].orientation.w)
            if i < len(msg.tolerances):
                tolerances.append(msg.tolerances[i].linear.x)
                tolerances.append(msg.tolerances[i].linear.y)
                tolerances.append(msg.tolerances[i].linear.z)
                tolerances.append(msg.tolerances[i].angular.x)
                tolerances.append(msg.tolerances[i].angular.y)
                tolerances.append(msg.tolerances[i].angular.z)
            else:
                for j in range(6):
                    tolerances.append(0.0)
        t0 = time.time()
        # print(positions)
        ik_solution = self.relaxed_ik.solve_position(positions, orientations, tolerances)
        # print(self.robot.articulated_joint_names)
        # print(ik_solution)
        # print(f"{(time.time() - t0)*1000:.2f}ms")
        # Publish the joint angle solution
        self.js_msg.header.stamp = rospy.Time.now()
        self.js_msg.position = ik_solution
        self.angles_pub.publish(self.js_msg)

    def pose_vels_cb(self, msg):
        linear_vels = []
        angular_vels = []
        tolerances = []
        for i in range(len(msg.ee_vels)):
            linear_vels.append(msg.ee_vels[i].linear.x)
            linear_vels.append(msg.ee_vels[i].linear.y)
            linear_vels.append(msg.ee_vels[i].linear.z)
            angular_vels.append(msg.ee_vels[i].angular.x)
            angular_vels.append(msg.ee_vels[i].angular.y)
            angular_vels.append(msg.ee_vels[i].angular.z)
            if i < len(msg.tolerances):
                tolerances.append(msg.tolerances[i].linear.x)
                tolerances.append(msg.tolerances[i].linear.y)
                tolerances.append(msg.tolerances[i].linear.z)
                tolerances.append(msg.tolerances[i].angular.x)
                tolerances.append(msg.tolerances[i].angular.y)
                tolerances.append(msg.tolerances[i].angular.z)
            else:
                for j in range(6):
                    tolerances.append(0.0)

        ik_solution = self.relaxed_ik.solve_velocity(linear_vels, angular_vels, tolerances)

        assert len(ik_solution) == len(self.robot.articulated_joint_names)

        # Publish the joint angle solution
        self.js_msg.header.stamp = rospy.Time.now()
        self.js_msg.position = ik_solution
        self.angles_pub.publish(self.js_msg)

    def marker_feedback_cb(self, msg):
        
        # Call the rust callback function
        self.relaxed_ik.dynamic_obstacle_cb(msg.marker_name, msg.pose)

    def marker_update_cb(self, msg):
        # update dynamic collision obstacles in relaxed IK
        for pose_stamped in msg.poses:
            # Call the rust callback function
            # print(pose_stamped.name, pos_arr[0],pos_arr[1],pos_arr[2])
            self.relaxed_ik.dynamic_obstacle_cb(pose_stamped.name, pose_stamped.pose)
    
    def ik_update_weight_cb(self, msg):
        self.update_objective_weights({
            msg.weight_name : msg.value
        })
    
    def update_objective_weights(self, weights_dict: dict):
        if not weights_dict:
            return 
        print(weights_dict)
        
        for k in weights_dict:
            if k not in self.weight_names:
                raise KeyError(k)
        for i in range(len(self.weight_names)):
            weight_name = self.weight_names[i]
            if weight_name in weights_dict:
                self.weight_priors[i] = weights_dict[weight_name]
        self.relaxed_ik.set_objective_weight_priors(self.weight_priors)
            
    
    
if __name__ == '__main__':
    rospy.init_node('relaxed_ik')
    print("RELAXED_IK_RUST.PY")
    relaxed_ik = RelaxedIK()
    # relaxed_ik.relaxed_ik.set_env_collision_tip_offset(0)
    # relaxed_ik.update_objective_weights({
    #     'eepos_4' : 0.0,
    #     'eequat_4' : 0.0,
    # })
    # enforce_joint_angles = [-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,  -10.0, 0.9, 0.9, 0.9, 1.5, 0.2, 0.0, 2.0, -2.0, 1.2354, 1.1, 0.9, 0.9, 0.9]
    # relaxed_ik.relaxed_ik.update_enforce_joint_angles(enforce_joint_angles)
    
    print(relaxed_ik.relaxed_ik.get_objective_weight_names())
    print(relaxed_ik.relaxed_ik.get_objective_weight_priors())
    rospy.spin()
