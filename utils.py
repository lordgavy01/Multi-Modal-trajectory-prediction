import numpy as np
from math import *
import pygame
import cv2
import heapq


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

image_name='geomap.png'
Background=Background(image_name,[0,0])
image=cv2.imread(image_name)
mapImage=image.copy()

grid=[[1]*Background.image.get_width() for _ in range(Background.image.get_height())]

for i in range(Background.image.get_height()):
    for j in range(Background.image.get_width()):
        if Background.image.get_at((j,i))[0]==0:
            grid[i][j]=0

Polygons=[]
radius_obstacle=30
# Read list of Polygons from out.txt where one line in that file is list of points of one polygon
with open('out.txt') as f:
    for line in f:
        Center=line.replace('\n','').split(' ')
        if Center==['']:
            continue
        Polygons.append((int(Center[0]),int(Center[1])))
