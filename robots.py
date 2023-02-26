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
        if d > r0 + r1:
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
        while not self.check_safe((x,y),obstacles_list,radius_obstacle):
            x = np.random.randint(xBound)
            y = np.random.randint(yBound)
        return (x, y)
    def steer(self,old_config,new_config,step_size):
        direction=atan2(new_config[1]-old_config[1],new_config[0]-old_config[0])
        q_config=(old_config[0]+step_size*cos(direction),old_config[1]+step_size*sin(direction))
        return q_config
    
class Manipulator:
    def __init__(self,base_position=(0,0), link_lengths=[15.0, 20.0, 15.0],pose=[0.0,0.0,0.0],joints_limit=pi/2):
        self.pose=pose
        self.base_position=base_position
        self.link_lengths = link_lengths
        self.joint_limits = joints_limit
    def check_safe(self,config,obstacles_list,radius_obstacle):
        links_rectangles=self.get_rectangle(config)
        for obstacle in obstacles_list:
            for link in links_rectangles:
                if self.check_circle_rect_intersection(link,obstacle,radius_obstacle):
                    return False 
        return True
    def check_circle_rect_intersection(self,link,obstacle,radius_obstacle):
        for i in range(len(link)):
            Pta=link[i]
            Ptb=link[(i+1)%len(link)]
            if self.check_circle_line_intersection((Pta,Ptb),obstacle,radius_obstacle):
                return True
        return False  
    def check_circle_line_intersection(self,line,obstacle,radius_obstacle):
        (p1x, p1y), (p2x, p2y), (cx, cy) = line[0], line[1], obstacle
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = radius_obstacle ** 2 * dr ** 2 - big_d ** 2

        return (discriminant>=0)
    def get_rectangle(self,config):
        Poly=[]
        Base=self.base_position
        lengths=self.link_lengths
        P1=(Base[0]+lengths[0]*cos(config[0]),Base[1]+lengths[0]*sin(config[0]))
        # Make a variable mid that is midpoint of Base and P1
        Mid=((Base[0]+P1[0])/2,(Base[1]+P1[1])/2)
        C1,C2,C3,C4=self.get_corners(config[0],Mid[0],Mid[1],lengths[0],lengths[0])
        Poly.append([C1,C4,C3,C2])
        old_theta=0
        old_point=P1
        
        for i in range(1,len(config)):
            old_theta+=config[i-1]
            new_theta=config[i]
            cur_point=(old_point[0]+lengths[i]*cos(new_theta+old_theta),old_point[1]+lengths[i]*sin(new_theta+old_theta))
            Mid=((old_point[0]+cur_point[0])/2,(old_point[1]+cur_point[1])/2)
            C1,C2,C3,C4=self.get_corners(new_theta+old_theta,Mid[0],Mid[1],lengths[i],lengths[i])
            Poly.append([Mid,[C1,C4,C3,C2]])
            old_point=cur_point
        return Poly
    
    def interpolate(self,config_1,config_2,l):
        new_config=[]
        for i in range(len(config_1)):
            new_c=config_1[i]*l+config_2[i]*(1-l)
            new_config.append(new_c)
        return new_config
    def generate_random_config(self,xBound,yBound,obstacles_list,radius_obstacle):
        new_config=[np.random.uniform(3*pi/4) for _ in range(3)]
        while not self.check_safe(new_config,obstacles_list,radius_obstacle):
            new_config=[np.random.uniform(3*pi/4) for _ in range(3)]
        return new_config
    def get_corners(theta,X,Y,L,B): 
        l=L/2
        b=B/2
        C1=(X+l*cos(theta)+b*cos(pi/2+theta),Y+l*sin(theta)+b*sin(pi/2+theta))
        C2=(X-l*cos(theta)+b*cos(pi/2+theta),Y-l*sin(theta)+b*sin(pi/2+theta))
        C3=(X-l*cos(theta)-b*cos(pi/2+theta),Y-l*sin(theta)-b*sin(pi/2+theta))
        C4=(X+l*cos(theta)-b*cos(pi/2+theta),Y+l*sin(theta)-b*sin(pi/2+theta))
        return C1,C2,C3,C4


    def steer(self,old_config,new_config,step_size):
        pass
    