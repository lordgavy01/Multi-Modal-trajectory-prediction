import numpy as np
import random
from robots import *
from collections import defaultdict

class RRTStar:
    def __init__(self, robot_type, start, goal,obstacles_list,radius_obstacle,occupancy_map=None, max_iterations=1000, step_size=0.1, goal_tolerance=10, rewiring_radius=25.0):
        self.robot_type = robot_type
        self.start = start
        self.goal = goal
        self.occupancy_map = occupancy_map
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.rewiring_radius = rewiring_radius
        self.vertices = [start]
        self.parent_vertex ={}
        self.edges = []
        self.costs = defaultdict(int)
        self.obstacles=obstacles_list
        self.parent_vertex[0]=-1
        self.radius_obstacle=radius_obstacle
        
    def distance(self, q1, q2):
        # Compute Euclidean distance between two points in configuration space
        return np.linalg.norm(np.array(q1) - np.array(q2))
    
    def collision_free(self, q):
        return self.robot_type.check_safe(q,self.obstacles,self.radius_obstacle)
    
    def get_random_config(self,xBound,yBound):
        return self.robot_type.generate_random_config(xBound,yBound,self.obstacles,self.radius_obstacle)
    def add_vertex(self,qnew,qnear_index):
        self.parent_vertex[len(self.vertices)]=qnear_index
        self.vertices.append(qnew)
    def steer(self,old_config,new_config):
        return self.robot_type.steer(old_config,new_config,self.step_size)
    
    def nearest_vertex(self, q):
        # Find the nearest vertex in the tree to the given configuration
        distances = [self.distance(q, v) for v in self.vertices]
        nearest_idx = np.argmin(distances)
        return nearest_idx
    
    def collision_free_path(self,old_config,new_config,get_path=False,eps_step=0.25):
        L=0
        Path=[]
        while L<=1:
            mid_config=self.robot_type.interpolate(old_config,new_config,L)
            if get_path==False and not self.collision_free(mid_config):
                return False
            Path.append(mid_config)
            L+=eps_step
        if get_path==True:
            return Path
        return True
    
    def get_path_to_goal(self):
        Path=[]
        node=self.nearest_vertex(self.goal)
        while self.parent_vertex[node]!=-1:
            Path.append(self.vertices[node])
            node=self.parent_vertex[node]
        return Path
    
    def rewire(self, q):
        # Check if any neighbors of q can be connected to it with a shorter path
        neighbors = [i for i,v in enumerate(self.vertices) if self.distance(q, v) <= self.rewiring_radius]
        q_index=len(self.vertices)-1
        for neighbor in neighbors:
            new_cost = self.costs[q_index] + self.distance(q, self.vertices[neighbor])
            if new_cost < self.costs[neighbor]:
                # Remove the old edge and add the new one
                # self.edges.remove((self.costs[neighbor], neighbor, neighbor.parent))
                # self.edges.append((new_cost, neighbor, q))
                # Update the cost and parent of the neighbor
                self.costs[neighbor] = new_cost
                self.parent_vertex[neighbor]= q

        