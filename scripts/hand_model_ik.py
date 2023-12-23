import ctypes
import numpy as np
import os, time, sys
import sys
import transformations as T
import yaml

from urdf_parser_py.urdf import URDF
from kdl_parser import kdl_tree_from_urdf_model
import PyKDL as kdl
from robot import Robot
import argparse
from hand_model import HandModel, load_object


# make python find the package
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/../')
from relaxed_ik_core.wrappers.python_wrapper import RelaxedIKRust, lib




class RelaxedIKDemo:
    def __init__(self, path_to_src, robot_name=None, obj_name=None):

        if obj_name is None or robot_name is None:
            setting_file_path = path_to_src + '/configs/settings.yaml'
        else:
            setting_file_path = path_to_src + f'/configs/settings_{robot_name}_{obj_name}.yaml'

        # os.chdir(path_to_src)

        # Load the infomation
        
        print("setting_file_path: ", setting_file_path)
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
       
      
        self.robot = Robot(setting_file_path, path_to_src, use_ros=False)
        print(f"Robot Articulated Joint names: {self.robot.articulated_joint_names}")
        
        print('\nInitialize Solver...\n')
        self.relaxed_ik = RelaxedIKRust(setting_file_path)
        
        ##DEBUG
        self.setting_file_path = setting_file_path

        if 'starting_config' not in settings:
            settings['starting_config'] = [0.0] * len(self.robot.articulated_joint_names)
        else:
            assert len(settings['starting_config']) == len(self.robot.articulated_joint_names), \
                    "Starting config length does not match the number of joints"
           
        
        self.weight_names  = self.relaxed_ik.get_objective_weight_names()
        self.weight_priors = self.relaxed_ik.get_objective_weight_priors()
        print(self.weight_names)
        print(self.weight_priors)
        
        print(len([1 for x in self.weight_names if 'selfcollision' in x]), 'self collision pairs')
        print("\nSolver RelaxedIK initialized!\n")

    def reload_ik_rust(self):
        self.relaxed_ik.__exit__(None, None, None)
        del self.relaxed_ik
        self.relaxed_ik = RelaxedIKRust(self.setting_file_path)
        
    def get_ee_pose(self):
        ee_poses = self.relaxed_ik.get_ee_positions()
        ee_poses = np.array(ee_poses)
        ee_poses = ee_poses.reshape((len(ee_poses)//6, 6))
        ee_poses = ee_poses.tolist()
        return ee_poses

    def reset_cb(self, msg):
        n = len(msg.position)
        x = (ctypes.c_double * n)()
        for i in range(n):
            x[i] = msg.position[i]
        self.relaxed_ik.reset(x)
    
    def reset(self, joint_angles):
        n = len(joint_angles)
        x = (ctypes.c_double * n)()
        for i in range(n):
            x[i] = joint_angles[i]
        return self.relaxed_ik.reset(x)
    
    def query_loss(self, joint_angles):
        n = len(joint_angles)
        x = (ctypes.c_double * n)()
        for i in range(n):
            x[i] = joint_angles[i]
        return self.relaxed_ik.get_jointstate_loss(x)

    def solve_pose_goals(self, positions, orientations, tolerances):
        # t0 = time.time()
        ik_solution = self.relaxed_ik.solve_position(positions, orientations, tolerances)
        # print(self.robot.articulated_joint_names)
        # print(ik_solution)
        # print(f"{(time.time() - t0)*1000:.2f}ms")
        return ik_solution
    
    
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
                self.weight_priors[i] = float(weights_dict[weight_name])
        self.relaxed_ik.set_objective_weight_priors(self.weight_priors)
            
    
    
if __name__ == '__main__':
    path_to_src = os.path.dirname(os.path.abspath(__file__)) + '/../relaxed_ik_core'
    print(path_to_src)
    
    
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-o', '--object', type=str)
    parser.add_argument('-r', '--recover', action='store_true')
    args = parser.parse_args()
    
    # relaxed_ik = RelaxedIKDemo(path_to_src, robot_name='allegro', obj_name=args.object)
    # N = number of end effectors. N = 2 in this example
    # positions: 3*N 
    # orientations: 4*N (quaternions)
    # tolerances: 6*N
    
    # x1 = np.array([0.039331, 0.091392, -0.057363])  # index
    # x2 = np.array([0.063934, 0.091392, 0.027384])   # thumb
    # x3 = x2 + np.array([0.0, 0.05, 0.0])            # thumb new
    # x4 = x1 + np.array([0.0, -0.05, 0.0])           # middle
    if args.object == 'cup':
        x1 = np.array([0.039331, 0.091392, -0.057363])
        x2 = np.array([0.063934, 0.091392, 0.027384])
        # x3 = x2 + np.array([0.0, 0.05, 0.0])
        r = (x1[0] ** 2 + x1[2] ** 2) ** 0.5
        theta = 1.0
        x3 = np.array([r * np.cos(theta), x1[1] + 0.05, r * np.sin(theta)])
        x4 = x1 + np.array([0.0, -0.05, 0.0])
        
    if args.object == '0':
        x1 = [ 0.1787013,  -0.01990971,  0.03727386]
        x2 = [ 0.15622276,  0.09828041, -0.0304384 ]
        x4 = [ 0.1617148,  -0.07427403, -0.02942346]
        x1 = np.array(x1)
        x2 = np.array(x2)
        x4 = np.array(x4)
        x3 = x2 + np.array([-0.03, 0.05, 0])
        
    if args.object == '6':
        x1 = [-0.10111307,  0.00385639, -0.05365375] 
        x2 = [-0.09719813,  0.04301281, -0.04002612]
        x4 = [-0.18000068, -0.12111433, -0.03776541]
        
        x1 = np.array(x1)
        x2 = np.array(x2)
        x4 = np.array(x4)
        x3 = x2 + np.array([0.03, 0.05, 0])
    
    if args.object == '40':
        
        x1 =  [ 0.05470081, -0.01059051, -0.02955591]
        x2 =  [-0.0101133,  -0.02552216, -0.13061427]
        x4 =  [ 0.08111349, -0.00227838, -0.03931702]
        
        x1 = np.array(x1)
        x2 = np.array(x2)
        x4 = np.array(x4)
        x3 = x2 + np.array([0, 0, 0.05])
        
    positions = list(x1) + list(x2) + list(x4)       # x0 y0 z0 x1 y1 z1
    
    orientations = [0.0, 0.0 ,0.0, 1.0, 
                    0.0, 0.0 ,0.0, 1.0,
                    0.0, 0.0 ,0.0, 1.0,]   # x0 y0 z0 w0 x1 y1 z1 w1
    tolerances = [0,0,0,0,0,0,
                  0,0,0,0,0,0,
                  0,0,0,0,0,0,]                    
    
    run_times = []
    recover_times = []
    
    
    urdf_path = '/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/relaxed_ik_core/configs/urdfs/allegro_hand.urdf'
    obj_path = '/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/scripts/obj_examples'
    
    joint_angle_list = []
    
    hand_model = HandModel(urdf_path, ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip'], low=0.01)
    pc = load_object(obj_path, hand_model, args.object)
    
    t0 = time.time()
    hand_model.ik([x1, x2, x4], pc, lr=50, max_iter=10000)
    ja = hand_model.getJoints()
    joint_angle_list.append(ja)
    t1 = time.time()
    result = hand_model.recover(pc, lr=50, max_iter=100)
    ja = hand_model.getJoints()
    joint_angle_list.append(ja)
    t2 = time.time()
    run_times.append(t1 - t0)
    recover_times.append(t2 - t1)
    
    # switch ee positions
    # positions = list(x1) + list(x3) + list(x4)
    q_list = []
    states = hand_model.move_finger(pc, q_list, ['link_15.0_tip'], np.array([x3]), lr=10, max_iter=1000)
    t3 = time.time()
    move_time = (t3 - t2) * 1000
    # print(len(q_list))
    # allegro.gui.loopStates(states)
    vis_idx = list(range(0, len(q_list), 100)) + ([] if (len(q_list) - 1) % 100 == 0 else [len(q_list) - 1])
    for i in vis_idx:
        joint_angle_list.append(q_list[i])
    
  
    sum_rik = sum(run_times)* 1000
    sum_recover = sum(recover_times)* 1000
    print(f"Total IK time: {sum(run_times)* 1000: .3f} ms. \nAverage IK time: {sum(run_times) / len(run_times) * 1000: .3f} ms.")
    if recover_times:
        print(f"Total Recover time:  {sum(recover_times)* 1000: .3f} ms. \nAverage Recover time: {sum(recover_times) / len(recover_times) * 1000: .3f} ms.")
    print(f"{sum_rik: .3f}+{sum_recover: .3f}+{move_time: 3f}={sum_rik + sum_recover+move_time: .3f}ms")
    
    
    np.save(f'/home/madcreeper/ik_recover/yg_grasp_traj_{args.object}.npy', np.array(joint_angle_list))
    np.save(f'/home/madcreeper/ik_recover/yg_grasp_points_{args.object}.npy', np.vstack([x1, x2, x3, x4]))
    
    
    
    