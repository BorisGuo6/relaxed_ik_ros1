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
            setting_file_path = path_to_src + f'/configs/grasp_yaml/{obj_name}.yaml'

        # os.chdir(path_to_src)

        # Load the infomation
        
        print("setting_file_path: ", setting_file_path)
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
        self.settings = settings
      
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
            
    
    


def main():
    path_to_src = os.path.dirname(os.path.abspath(__file__)) + '/../relaxed_ik_core'
    print(path_to_src)
    
    
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-o', '--object', type=str)
    parser.add_argument('-r', '--recover', action='store_true')
    args = parser.parse_args()
    
    relaxed_ik = RelaxedIKDemo(path_to_src, robot_name='allegro', obj_name=args.object)
    # N = number of end effectors. N = 2 in this example
    # positions: 3*N 
    # orientations: 4*N (quaternions)
    # tolerances: 6*N
    
    # x1 = np.array([0.039331, 0.091392, -0.057363])  # index
    # x2 = np.array([0.063934, 0.091392, 0.027384])   # thumb
    # x3 = x2 + np.array([0.0, 0.05, 0.0])            # thumb new
    # x4 = x1 + np.array([0.0, -0.05, 0.0])           # middle
    # if args.object == 'cup':
    #     x1 = np.array([0.039331, 0.091392, -0.057363])
    #     x2 = np.array([0.063934, 0.091392, 0.027384])
    #     # x3 = x2 + np.array([0.0, 0.05, 0.0])
    #     r = (x1[0] ** 2 + x1[2] ** 2) ** 0.5
    #     theta = 1.0
    #     x3 = np.array([r * np.cos(theta), x1[1] + 0.05, r * np.sin(theta)])
    #     x4 = x1 + np.array([0.0, -0.05, 0.0])
        
    # if args.object == '0':
    #     x1 = [ 0.1787013,  -0.01990971,  0.03727386]
    #     x2 = [ 0.15622276,  0.09828041, -0.0304384 ]
    #     x4 = [ 0.1617148,  -0.07427403, -0.02942346]
    #     x1 = np.array(x1)
    #     x2 = np.array(x2)
    #     x4 = np.array(x4)
    #     x3 = x2 + np.array([-0.03, 0.05, 0])
        
    # if args.object == '6':
    #     x1 = [-0.10111307,  0.00385639, -0.05365375] 
    #     x2 = [-0.09719813,  0.04301281, -0.04002612]
    #     x4 = [-0.18000068, -0.12111433, -0.03776541]
        
    #     x1 = np.array(x1)
    #     x2 = np.array(x2)
    #     x4 = np.array(x4)
    #     x3 = x2 + np.array([0.03, 0.05, 0])
    
    # if args.object == '40':
        
    #     x1 =  [ 0.05470081, -0.01059051, -0.02955591]
    #     x2 =  [-0.0101133,  -0.02552216, -0.13061427]
    #     x4 =  [ 0.08111349, -0.00227838, -0.03931702]
        
    #     x1 = np.array(x1)
    #     x2 = np.array(x2)
    #     x4 = np.array(x4)
    #     x3 = x2 + np.array([0, 0, 0.05])
        
    # positions = list(x1) + list(x2) + list(x4)       # x0 y0 z0 x1 y1 z1
    
    settings = relaxed_ik.settings
    is_active_chain = settings['is_active_chain']
    num_active_chains = is_active_chain.count(True)
    grasp_points_old = settings['grasp_points_old']
    grasp_points_new = settings['grasp_points_new']
    
    fingers_moved = []
    for i in range(len(grasp_points_old)):
        if grasp_points_old[i] and grasp_points_new[i]:
            if np.linalg.norm(np.array(grasp_points_new[i]) - np.array(grasp_points_old[i])) > 1e-3:
                fingers_moved.append(True)
                continue
        fingers_moved.append(False)
    
    assert fingers_moved.count(True) == 1
    
    moving_finger_idx = fingers_moved.index(True)
    
    all_fingers = [i for i, is_active in enumerate(is_active_chain)]
    all_active_fingers = [i for i, is_active in enumerate(is_active_chain) if is_active]
    nomove_fingers = all_active_fingers[:]
    nomove_fingers.remove(moving_finger_idx)
    
    positions = [x for point in grasp_points_old for x in point]
    
    
    
    orientations = [0.0, 0.0 ,0.0, 1.0] * num_active_chains
    tolerances = [0,0,0,0,0,0] * num_active_chains            
    
    run_times = []
    recover_times = []
    
    
    # urdf_path = '/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/relaxed_ik_core/configs/urdfs/allegro_hand.urdf'
    # obj_path = '/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/scripts/obj_examples'
    # hand_model = HandModel(urdf_path, ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip'], low=0.01)
    
    
    
    relaxed_ik.update_objective_weights({
        f"eepos_{i}": 100.0 for i in all_active_fingers 
    })
    joint_angle_list = []    
    
    # pc = load_object(obj_path, hand_model, args.object)
    assert(args.recover is True or args.recover is False)
    
    def execute_grasp_ik(positions, orientations, tolerances, pcd_w_alter, pcd_w_normal, ee_w_alter, ee_w_normal, recover=False):
        assert len(pcd_w_alter) == len(ee_w_alter)
        N = len(pcd_w_alter)
        for i in range(N):
            print("-"*50)
            t0 = time.time()
            
            w_dict_pcd = {}
            for j in all_fingers:
                pcdw = pcd_w_normal[i] if isinstance(pcd_w_normal, list) else pcd_w_normal
                w_dict_pcd[f"envcollision_pcd_{j}"] = pcdw
            w_dict_pcd[f"envcollision_pcd_{moving_finger_idx}"] = pcd_w_alter[i]
            relaxed_ik.update_objective_weights(w_dict_pcd)
            
            w_dict_ee = {}
            for j in all_active_fingers:
                w_dict_ee[f"eepos_{j}"] = ee_w_normal
            w_dict_ee[f"eepos_{moving_finger_idx}"] = ee_w_alter[i]
            relaxed_ik.update_objective_weights(w_dict_ee)

            ja = relaxed_ik.solve_pose_goals(positions, orientations, tolerances)
            print('loss:', relaxed_ik.query_loss(ja))  
            t1 = time.time()
            
            # if recover and args.recover:
            #     hand_model.setJoints(ja)
            #     result = hand_model.recover(pc, lr=50, max_iter=100)
            #     ja = list(hand_model.getJoints())
            #     print("recover result:", result)
            
            print("Joint Angles:", ja)
            joint_angle_list.append(ja)
            
            t2 = time.time()
            run_times.append(t1 - t0)
            recover_times.append(t2 - t1)
            print(f"RangedIK time: {(t1 - t0)*1000}ms")
            print(f"Recover time:  {(t2 - t1)*1000}ms")
            print("-"*50)
    
    
    t_pre0 = time.time()
    # pre-grasp
    # pcd_w = [(w,w,w) for w in [20, 20, 20, 20]]
    # ee_w  = [(100,100,100) for _ in range(len(pcd_w))]
    execute_grasp_ik(positions, orientations, tolerances, [20, 10, 5, 5], [20, 10, 5, 5], [100]*4, 100, False)
    
    # grasp
    # pcd_w = [(w,w,w) for w in [0.1, 0.1, 0.1]]
    # ee_w  = [(100,100,100) for _ in range(len(pcd_w))]
    relaxed_ik.relaxed_ik.set_fix_ee_indices(all_active_fingers)
    execute_grasp_ik(positions, orientations, tolerances, [0.1]*4, 0.1, [100]*4, 100, True)
    
    
    # switch ee positions
    # positions = list(x1) + list(x3) + list(x4)
    positions = [x for point in grasp_points_new for x in point]
    
    t_pre1 = time.time()
    # release
    # pcd_w = [(0.5,w,0.5) for w in [1, 10, 10]]
    # ee_w  = [(100,1,100) for _ in range(8)]
    relaxed_ik.relaxed_ik.set_fix_ee_indices(nomove_fingers)
    execute_grasp_ik(positions, orientations, tolerances,[1, 10, 10], 0.5, [1, 1, 1], 100, True)
    
    # 2nd pre-grasp
    # pcd_w = [(0.5,w,0.5) for w in [10, 10, 10]]
    # ee_w  = [(100,100,100) for _ in range(len(pcd_w))]
    relaxed_ik.relaxed_ik.set_fix_ee_indices(nomove_fingers)
    execute_grasp_ik(positions, orientations, tolerances,[10, 10, 10], 0.5, [100, 100, 100], 100, True)
    
    # grasp again
    # pcd_w = [(w,w,w) for w in [0.1, 0.1, 0.1]]
    # ee_w  = [(100,100,100) for _ in range(len(pcd_w))]
    relaxed_ik.relaxed_ik.set_fix_ee_indices(all_active_fingers)
    execute_grasp_ik(positions, orientations, tolerances, [0.1,0.1, 0.1], 0.1, [100, 100, 100], 100, True)
    
    t_after1 = time.time()
    # pcd_w = [(0.5, w, 0.5) for w in [0.5, 0.5]]
    # ee_w  = [(100,100,100) for _ in range(len(pcd_w))]
    # # relaxed_ik.relaxed_ik.set_fix_ee_indices([0, 1, 2])
    # execute_grasp_ik(positions, orientations, tolerances, pcd_w, ee_w)
    
    sum_rik = sum(run_times)* 1000
    sum_recover = sum(recover_times)* 1000
    
    print(f"Total RangedIK time: {sum(run_times)* 1000: .3f} ms. \nAverage RangedIK time: {sum(run_times) / len(run_times) * 1000: .3f} ms.")
    if recover_times:
        print(f"Total Recover time:  {sum(recover_times)* 1000: .3f} ms. \nAverage Recover time: {sum(recover_times) / len(recover_times) * 1000: .3f} ms.")
    print(f"{sum_rik: .3f}+{sum_recover: .3f}={sum_rik + sum_recover: .3f}ms")
    np.save(f'/home/madcreeper/ik_recover/grasp_traj/grasp_traj_{args.object}.npy', np.array(joint_angle_list))
    
    with open('/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/scripts/time_log.txt', 'a') as f:
        f.write(f"{t_pre1 - t_pre0},{t_after1-t_pre1}\n")
    
    # np.save(f'/home/madcreeper/ik_recover/grasp_points_{args.object}.npy', np.vstack([x1, x2, x3, x4]))
    
    
    
if __name__ == '__main__':
    main()