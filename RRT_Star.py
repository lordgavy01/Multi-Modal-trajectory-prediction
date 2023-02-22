import numpy as np
import random
from robots import *

class RRTStar:
    def __init__(self, robot_type, start, goal, occupancy_map, max_iterations=1000, step_size=0.1, goal_tolerance=0.5, rewiring_radius=1.0):
        self.robot_type = robot_type
        self.start = start
        self.goal = goal
        self.occupancy_map = occupancy_map
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.rewiring_radius = rewiring_radius
        self.vertices = [start]
        self.edges = []
        self.costs = {self.start: 0}
        
    def distance(self, q1, q2):
        # Compute Euclidean distance between two points in configuration space
        return np.linalg.norm(np.array(q1) - np.array(q2))
    
    def collision_free(self, q):
        return self.robot_type.check_safe(self.occupancy_map,q)
    def get_random_config(self):
        return self.robot_type.generate_random_config()
    
    def steer(self,old_config,new_config):
        return self.robot_type.move(old_config,new_config,self.step_size)
    
    def nearest(self, q):
        # Find the nearest vertex in the tree to the given configuration
        distances = [self.distance(q, v) for v in self.vertices]
        nearest_idx = np.argmin(distances)
        return self.vertices[nearest_idx]
    
    def collision_free_path(self,old_config,new_config,get_path=False,eps_step=0.1):
        L=0
        Path=[]
        while L<=1:
            mid_config=self.robot_type.interpolate(old_config,new_config,L)
            if get_path==False and not self.collision_free(mid_config):
                return False
            Path.append(mid_config)
            L+=eps_step
        return Path

    def rewire(self, q):
        # Check if any neighbors of q can be connected to it with a shorter path
        neighbors = [v for v in self.vertices if self.distance(q, v) <= self.rewiring_radius]
        for neighbor in neighbors:
            new_cost = self.costs[q] + self.distance(q, neighbor)
            if new_cost < self.costs[neighbor]:
                # Remove the old edge and add the new one
                self.edges.remove((self.costs[neighbor], neighbor, neighbor.parent))
                self.edges.append((new_cost, neighbor, q))
                # Update the cost and parent of the neighbor
                self.costs[neighbor] = new_cost
                neighbor.parent = q

           
