import numpy as np
from math import *

class MobileRobot:
    def __init__(self, pose=(0,0,pi/2), radius=5):
        self.pose = pose
        self.track = radius
    def check_safe(self,config,occupancy_map):
        return True
    def collision_free(self,source_config,goal_config,occupancy_map)
        pass
    def generate_random_config(self,xBound,yBound,angleBound):
        x = np.random.uniform(xBound)
        y = np.random.uniform(yBound)
        theta=np.random.uniform(angleBound)
        while self.check_safe((x,y,theta)):
            x = np.random.uniform(xBound)
            y = np.random.uniform(yBound)
            theta=np.random.uniform(angleBound)
        return (x, y,theta)
    def steer(self,old_config,new_config,step_size):
        direction=atan2(new_config[1]-old_config[1],new_config[0]-old_config[0])
        q_config=(old_config[0]+step_size*cos(direction),old_config[1]+step_size*sin(direction),old_config[2]+np.random.uniform(pi/4))
        return q_config
    
class Manipulator:
    def __init__(self,base_position=(0,0), link_lengths=[1.0, 1.0, 1.0],pose=[0.0,0.0,0.0],joints_limit=pi/2):
        self.pose=pose
        self.base_position=base_position
        self.link_lengths = link_lengths
        self.joint_limits = joints_limit
    def check_safe(self,config,occupancy_map):
        return True
    def collision_free(self,source_config,goal_config,occupancy_map):
        pass