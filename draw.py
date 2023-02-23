from time import sleep
import pygame
import math
# make pygame window with the functionality to draw heaxagons wherever the mouse is clicked
# make a function that draws a hexagon at the mouse position
pygame.init()

screen=pygame.display.set_mode((500,500))
running=True


Poly=[]
def draw_obstacle(screen,x,y,side_length):
    global Poly
    col=(0,0,0)
    pygame.draw.circle(screen,col,(x,y),side_length)
    Poly.append((x,y))

green=(0,255,0)
flag=0
font = pygame.font.Font('freesansbold.ttf', 32)
screen.fill((255,255,255))
while running:

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        if event.type==pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            x,y=pos
            side_length=30
            draw_obstacle(screen,x,y,side_length)
            # pygame.draw.polygon(screen,(255,0,0),[(x,y),(x+100,y-50),(x+100,y+50),(x,y+100),(x-100,y+50),(x-100,y-50)])
        
    pygame.display.flip()

pygame.image.save(screen,"geomap.png")
# write list of points in Poly to file out.txt
with open("out.txt","w") as f:
    for i in Poly:
        f.write(str(i[0])+" "+str(i[1]))
        f.write("\n")
    f.write("\n")
print(Poly)