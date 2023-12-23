from scipy.spatial.transform import Rotation as R
import nimblephysics as nimble
# import pybullet as p
import cvxpy as cp
import numpy as np
import torch
from tqdm import trange
from collision_object import *
from time import time
import argparse
from torch.optim import Adam
# from distance import distance_box


class Link():
    def __init__(self, node):
        self.node = node
        self.name = node.getName()
        total_dof = node.getSkeleton().getNumDofs()
        self.jacobian = np.zeros((3, total_dof))
        self.update()
    
    def update(self):
        self.position = self.node.getTransform().translation()
        dep_jacobian = self.node.getWorldJacobian([0.0]*3)
        dep_dof = self.node.getNumDependentGenCoords()
        for i in range(dep_dof):
            idx = self.node.getDependentGenCoordIndex(i)
            self.jacobian[:, idx] = dep_jacobian[3:, i]


class HandModel():
    def __init__(self, urdf_path, tips, low=0.03, high=0.08):
        self.world = nimble.simulation.World()
        self.world.setTimeStep(0.01)
        # self.gui = nimble.NimbleGUI(self.world)
        # self.gui.serve(8089)

        self.hand = self.world.loadSkeleton(urdf_path)
        self.tips = tips
        self.num_tips = len(tips)

        #self.rest_pose = np.array((0, -1.5, 0., -0., 0., -0.5,
                                #0, 0, 0, 0, 0., 0, 0, 0,
                                #0, 0, 0, 0, 0, 0, 0, 0))
                                
        self.rest_pose = np.array((0, 0, 0., 0., 0., 0,
                                0, 0, 0, 0, 0., 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0))
        self.world.setPositions(self.rest_pose)

        self.links = {}
        for link in self.hand.getBodyNodes():
            self.links[link.getName()] = Link(link)
        # print(self.hand.getBodyNodes())
        self.low_distance = low
        self.high_distance = high

    def getJoints(self):
        return self.world.getPositions()
    
    def updateLinks(self):
        for _, link in self.links.items():
            link.update()
    
    def setJoints(self, q):
        action = torch.zeros((22,))
        self.world.setPositions(q)
        # state = torch.Tensor(self.world.getState())
        # state = nimble.timestep(self.world, state, action)
        self.updateLinks()

    def moveJoints(self, dq):
        action = torch.zeros((22,))
        self.world.setVelocities(dq)
        state = torch.Tensor(self.world.getState())
        state = nimble.timestep(self.world, state, action)
        self.updateLinks()
        return state
    
    def pc_sdf(self, pc: PointCloud, pos):
        indice, knn = pc.kNN(pos)
        p0 = knn[0]
        n0 = pc.normals[indice[0]]
        dp = p0 - pos
        sign = np.sign(np.dot(dp, n0))
        sdf = np.linalg.norm(dp) * sign

        if sdf > self.high_distance:
            return sdf, np.zeros(3)
        else:
            return sdf, -n0
        
    def ik_pos_loss(self, ik_pos_target):
        dis_norms = []
        for i in range(self.num_tips):
            pos_tip = self.links[self.tips[i]].position
            delta_tip = ik_pos_target[i] - pos_tip
            dis_norms.append(np.linalg.norm(delta_tip))
        return dis_norms

    def ik(self, target_tips: list, pc: PointCloud,lr = 50, max_iter=10000):
        for iter in trange(max_iter):
            grad_q = np.zeros(22)
            delta_tips = []
            for i in range(self.num_tips):
                tip = self.tips[i]
                J_tip = self.links[tip].jacobian
                pos_tip = self.links[tip].position
                delta_tip = target_tips[i] - pos_tip
                grad_q += delta_tip @ J_tip 
                delta_tips.append(delta_tip)
                # print(grad_q)

            # for name, link in self.links.items():
            #     if name in self.tips:
            #         continue
            #     J_link = link.jacobian
            #     pos_link = link.position
            #     sdf, delta_link = self.pc_sdf(pc, pos_link)
            #     grad_q += delta_link @ J_link
            self.moveJoints(lr * grad_q)
            #self.gui.nativeAPI().renderWorld(allegro.world, "world")
            delta_norm = np.linalg.norm(np.hstack(delta_tips))

            
            if delta_norm < 1e-4:
                for name, link in self.links.items():
                    pos_link = link.position
                    sdf, delta_link = self.pc_sdf(pc, pos_link)
                    print('Name', link.name)
                    print('Distance', sdf)
                return True
            

        return delta_norm
    
    def recover(self, pc: PointCloud, lr = 0.1, max_iter=5000):
        for iter in trange(max_iter):
            grad_q = np.zeros(22)
            distances = []
            for name, link in self.links.items():
                if name in self.tips:
                    continue
                J_link = link.jacobian
                pos_link = link.position
                sdf, delta_link = self.pc_sdf(pc, pos_link)
                grad_q += delta_link @ J_link
                distances.append(sdf)

            J_tips = []
            for i in range(self.num_tips):
                tip = self.tips[i]
                J_tip = self.links[tip].jacobian
                J_tips.append(J_tip)
            J_tips = np.vstack(J_tips)

            proj = np.linalg.pinv(J_tips) @ J_tips
            grad_q -= proj @ grad_q

            self.moveJoints(lr * grad_q)
            min_dist = min(distances)

            if min_dist > self.low_distance:
                for name, link in self.links.items():
                    pos_link = link.position
                    sdf, delta_link = self.pc_sdf(pc, pos_link)
                    print('Name', link.name)
                    print('Distance', sdf)
                return True

        return min_dist

    def move_finger(self, pc: PointCloud, q_list: list, move_tips: list, target_tips: np.ndarray, lr = 0.1, max_iter=5000, patience=2):
        fixed_tips = [tip for tip in self.tips if tip not in move_tips]
        print("fixed tips:", fixed_tips)
        print("target:", target_tips)
	    # render
        states = []
        
        losses = []
        
        for iter in trange(max_iter):
            # render
            # if iter % 100 == 0:
            #     states.append(torch.Tensor(self.world.getState()))

            grad_q = np.zeros(22)
            distances = []
            delta_tips = []
            for name, link in self.links.items():
                if name in fixed_tips:
                    continue
                J_link = link.jacobian
                pos_link = link.position
                sdf, delta_link = self.pc_sdf(pc, pos_link)
                grad_q += 0.001 * delta_link @ J_link
                distances.append(sdf)
            
            for i, tip in enumerate(move_tips):
                J_tip = self.links[tip].jacobian
                pos_tip = self.links[tip].position
                delta_tip = target_tips[i] - pos_tip
                grad_q += delta_tip @ J_tip 
                # print(pos_tip, delta_tip, target_tips)
                delta_tips.append(delta_tip)
                
            J_tips = []
            for tip in fixed_tips:
                J_tip = self.links[tip].jacobian
                J_tips.append(J_tip)
            J_tips = np.vstack(J_tips)
            
            
            proj = np.linalg.pinv(J_tips) @ J_tips
            grad_q -= proj @ grad_q

            dq = lr * grad_q
            self.moveJoints(dq)
            q_list.append(self.getJoints())
            
            delta_norm = np.linalg.norm(np.hstack(delta_tips))
            losses.append(delta_norm)
            
            if iter > patience and losses[-1 - patience] - losses[-1] < 1e-7:
                lr *= 0.5
                print(lr)
            
            # if delta_norm < 1e-4:
            #     for name, link in self.links.items():
            #         pos_link = link.position
            #         sdf, delta_link = self.pc_sdf(pc, pos_link)
            #         # print('Name', link.name)
            #         # print('Distance', sdf)
            #     return True
            
        return states
    
    


        
if __name__ == "__main__":
    urdf_path = '/home/madcreeper/rangedik_project/src/relaxed_ik_ros1/relaxed_ik_core/configs/urdfs/allegro_hand.urdf'
    allegro = HandModel(urdf_path, ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip'], low=0.035)

    # for name, link in allegro.links.items():
    #     print('Name', link.name)
    #     print('Position', link.position)
    # input()


    # num_points = 10000
    # points = generate_unit_sphere(num_points, radius=0.1)
    # pc = PointCloud(points, - points)

    mesh_path = './cup.obj'
    points = np.load('cup_points.npy')
    normals = np.load('cup_normals.npy')
    pc = PointCloud(points, normals)
    allegro.world.loadSkeleton("./cup.urdf")

    x1 = np.array([0.039331, 0.091392, -0.057363])
    x2 = np.array([0.063934, 0.091392, 0.027384])
    # x3 = x2 + np.array([0.0, 0.05, 0.0])
    r = (x1[0] ** 2 + x1[2] ** 2) ** 0.5
    theta = 2
    x3 = np.array([r * np.cos(theta), x1[1], r * np.sin(theta)])
    x4 = x1 + np.array([0.0, -0.05, 0.0])
    


    # print(allegro.links['link_3.0_tip'].position)
    # print(allegro.links['link_7.0_tip'].position)
    # print(allegro.links['link_15.0_tip'].position)  
