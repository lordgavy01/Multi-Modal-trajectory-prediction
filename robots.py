import numpy as np
from math import *

class MobileRobot:
    def __init__(self, pose=(0,0), radius=5):
        self.pose = pose
        self.radius = radius
    def check_safe(self,config,obstacles_list,radius_obstacle):
        for obstacle in obstacles_list:
            if self.check_circle_intersection(config,obstacle,radius_obstacle):
                return False 
        return True
    def interpolate(self,config_1,config_2,l):
        x=config_1[0]*l+config_2[0]*(1-l)
        y=config_1[1]*l+config_2[1]*(1-l)
        return (x,y)
    def check_circle_intersection(self,circle1,circle2,radius_obstacle):
        x0,y0=circle1
        x1,y1=circle2
        r0=self.radius
        r1=radius_obstacle
        d=sqrt((x1-x0)**2 + (y1-y0)**2)

        # non intersecting
        if d > r0 + r1 :
            return False
        # # One circle within other
        # if d < abs(r0-r1):
        #     return True
        # # coincident circles
        # if d == 0 and r0 == r1:
        #     return True
        # else:
        #     a=(r0**2-r1**2+d**2)/(2*d)
        #     h=sqrt(r0**2-a**2)
        #     x2=x0+a*(x1-x0)/d   
        #     y2=y0+a*(y1-y0)/d   
        #     x3=x2+h*(y1-y0)/d     
        #     y3=y2-h*(x1-x0)/d 
        #     x4=x2-h*(y1-y0)/d
        #     y4=y2+h*(x1-x0)/d
        #     return x3, y3, x4, y4
        return True
    def generate_random_config(self,xBound,yBound,obstacles_list,radius_obstacle):
        x = np.random.uniform(xBound)
        y = np.random.uniform(yBound)
        while self.check_safe((x,y),obstacles_list,radius_obstacle):
            x = np.random.uniform(xBound)
            y = np.random.uniform(yBound)
        return (x, y)
    def steer(self,old_config,new_config,step_size):
        direction=atan2(new_config[1]-old_config[1],new_config[0]-old_config[0])
        q_config=(old_config[0]+step_size*cos(direction),old_config[1]+step_size*sin(direction))
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