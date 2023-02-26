import numpy as np
from math import *
import pygame
import cv2
import heapq


# np.random.seed(12)
pygame.init()
class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

image_name='narrow.png'
Background=Background(image_name,[0,0])
length=40
breadth=8

# This function returns the corners of the rectangle
def get_corners(theta,X,Y): 
    l=length/2
    b=breadth/2
    C1=(X+l*cos(theta)+b*cos(pi/2+theta),Y+l*sin(theta)+b*sin(pi/2+theta))
    C2=(X-l*cos(theta)+b*cos(pi/2+theta),Y-l*sin(theta)+b*sin(pi/2+theta))
    C3=(X-l*cos(theta)-b*cos(pi/2+theta),Y-l*sin(theta)-b*sin(pi/2+theta))
    C4=(X+l*cos(theta)-b*cos(pi/2+theta),Y+l*sin(theta)-b*sin(pi/2+theta))
    return C1,C2,C3,C4

# This function returns the distance between two points with wt to adjust theta
def Euclidean(A,B,wt=50):
  return sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+wt*min((A[2]-B[2])**2,(2*pi-abs(A[2]-B[2]))**2))

grid=[[1]*Background.image.get_width() for _ in range(Background.image.get_height())]

for i in range(Background.image.get_height()):
    for j in range(Background.image.get_width()):
        if Background.image.get_at((j,i))[0]==0:
            grid[i][j]=0

Polygons=[]
# Read list of Polygons from out.txt where one line in that file is list of points of one polygon
with open('out.txt') as f:
    for line in f:
        Polygon=[]
        lst=[]
        comma=False
        Fir,Sec='',''
        Cent=(-1,-1)
        for p in line:
            if p==' ' or p=='(':
                continue
            if p==',':
                comma=True
                continue
            if p==')':
                if Cent[0]==-1:
                    Cent=(float(Fir),float(Sec))
                else:
                    lst.append((float(Fir),float(Sec)))
                Fir,Sec='',''
                comma=False
                continue
            if comma:
                Sec+=p
            else:
                Fir+=p              
        if len(lst):
            Polygon=[Cent,lst]
            Polygons.append(Polygon)


dir=[(1,0),(0,1),(-1,0),(0,-1),(0,0)]

def line_intersect(Points):
    Line1=Points[:2]
    Line2=Points[2:]
    dx0 = Line1[1][0]-Line1[0][0]
    dx1 = Line2[1][0]-Line2[0][0]
    dy0 = Line1[1][1]-Line1[0][1]
    dy1 = Line2[1][1]-Line2[0][1]
    p0 = dy1*(Line2[1][0]-Line1[0][0]) - dx1*(Line2[1][1]-Line1[0][1])
    p1 = dy1*(Line2[1][0]-Line1[1][0]) - dx1*(Line2[1][1]-Line1[1][1])
    p2 = dy0*(Line1[1][0]-Line2[0][0]) - dx0*(Line1[1][1]-Line2[0][1])
    p3 = dy0*(Line1[1][0]-Line2[1][0]) - dx0*(Line1[1][1]-Line2[1][1])
    return (p0*p1<=0) & (p2*p3<=0)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check(Point, poly):
    center=poly[0]
    PointList=poly[1]
    
    n = len(PointList)
    p1x, p1y = PointList[0]
    for i in range(1,n + 1):
        p2x, p2y = PointList[i % n]
        if ccw(Point, (p1x, p1y), (p2x, p2y))!=ccw(center, (p1x, p1y), (p2x, p2y)):
            return False
        p1x, p1y = p2x, p2y
    return True

def check_inside(Poly1, Poly2):
    flag=1
    for i in range(len(Poly1[1])):
        flag&=check(Poly1[1][i],Poly2) 
    return flag
    
def check_intersect(Poly1, Poly2):
    if check_inside(Poly1, Poly2) or check_inside(Poly2, Poly1):
        return "One is inside the other"

    for i in range(len(Poly1[1])):
        for j in range(len(Poly2[1])):
            Points=[Poly1[1][i],Poly1[1][(i+1)%len(Poly1[1])],Poly2[1][j],Poly2[1][(j+1)%len(Poly2[1])]]
            if line_intersect(Points):
                return "Yes"
    return "No"

def dont_intersect(Pol1,Pol2):
    # check if the bounding boxes of the two polygons overlap , if yes return false
    # else return true
    # Pol1 stores list of points of first polygon
    # Pol2 stores list of points of second polygon
    # Poly1 is the first polygon
    # Poly2 is the second polygon
    Poly1=Pol1[1]
    Poly2=Pol2[1]
    # Get the bounding box of the two polygons
    x1=min(Poly1,key=lambda x:x[0])[0]
    x2=max(Poly1,key=lambda x:x[0])[0]
    y1=min(Poly1,key=lambda x:x[1])[1]
    y2=max(Poly1,key=lambda x:x[1])[1]
    x3=min(Poly2,key=lambda x:x[0])[0]
    x4=max(Poly2,key=lambda x:x[0])[0]
    y3=min(Poly2,key=lambda x:x[1])[1]
    y4=max(Poly2,key=lambda x:x[1])[1]
    # Check if the bounding boxes overlap
    if x1>x4 or x2<x3 or y1>y4 or y2<y3:
        return True
    return False

def lineCollision(A,B):
    # A and B are the two points of the line
    # Check if the line intersects with any of the polygons
    for i in range(len(Polygons)):
        for j in range(len(Polygons[i][1])):
            Points=[A,B,Polygons[i][1][j],Polygons[i][1][(j+1)%len(Polygons[i][1])]]
            if line_intersect(Points):
                return True
    return False

def boundaryCheck(A):
    for i in range(-60,60):
        for j in range(-60,60):
            x=A[0]+i
            y=A[1]+j
            if x>=0 and x<Background.image.get_width() and y>=0 and y<Background.image.get_height():
                continue
            else:
                return False
    return True

def Coli(A,B):
    L=0
    eps=1/dist(A,B)
    while L<=1:
        R=(A[0]+L*(B[0]-A[0]),A[1]+L*(B[1]-A[1]))
        if R[0]>=0 and R[0]<Background.image.get_width() and R[1]>=0 and R[1]<Background.image.get_height():
            if grid[int(R[1])][int(R[0])]==0:
                return True
        else:
            return True
        L+=eps
    return False
# This function checks if the give configuration is safe or not
def check_safe(theta,X,Y):
    C1,C2,C3,C4=get_corners(theta,X,Y)
    Current=[(X,Y),[C1,C4,C3,C2]]
    if C1[0]>=0 and C1[1]>=0 and C2[0]>=0 and C2[1]>=0 and C3[0]>=0 and C3[1]>=0 and C4[0]>=0 and C4[1]>=0 and C1[0]<Background.image.get_width() and C1[1]<Background.image.get_height() and C2[0]<Background.image.get_width() and C2[1]<Background.image.get_height() and C3[0]<Background.image.get_width() and C3[1]<Background.image.get_height() and C4[0]<Background.image.get_width() and C4[1]<Background.image.get_height():
        if grid[int(C1[1])][int(C1[0])]==0 or grid[int(C2[1])][int(C2[0])]==0 or grid[int(C3[1])][int(C3[0])]==0 or grid[int(C4[1])][int(C4[0])]==0:
            return False
        for poly in Polygons:
            if dont_intersect(Current,poly):
                continue
            if check_intersect(Current,poly)!="No":
                return False
        return True
    return False


def check_feasible(A,B,givePath=False,eps=0.1):
    L=0
    if givePath==False:
        eps=1/Euclidean(A,B,0)
    # check if the line segment connecting A and B is feasible
    Flag1,Flag2=True,True
    Pts=[]
    thetas1=[]
    thetas2=[]
    while L<=1:
        # using A and B to find the point Pt
        Pt=(A[0]*L+B[0]*(1-L),A[1]*L+B[1]*(1-L))
        Pts.append(Pt)
        theta1=(A[2]*L+B[2]*(1-L))
        theta2=(A[2]*L+(-2*pi+B[2])*(1-L))
        thetas1.append(theta1)
        thetas2.append(theta2)
        # check if the point is safe
        if check_safe(theta1,Pt[0],Pt[1])==False:
            Flag1=False
        if check_safe(theta2,Pt[0],Pt[1])==False:
            Flag2=False
        
        L+=eps

    if not givePath:
        if Flag1==True or Flag2==True:
            return True

        return False
    else:
        prefer=1
        if Flag1==True and Flag2==True:
            if abs(B[2]-A[2])>abs(2*pi+A[2]-B[2]):
                prefer=2
        elif Flag2==True:
            prefer=2
        Path=[]
        if prefer==1:
            for i in range(len(Pts)):
                Path.append((Pts[i][0],Pts[i][1],thetas1[i]))
        else:
            for i in range(len(Pts)):
                Path.append((Pts[i][0],Pts[i][1],thetas2[i]))
        return Path

configPoints=[]
def getSourceGoal():
    screen = pygame.display.set_mode((Background.image.get_width(),Background.image.get_height()))
    screen.blit(Background.image, Background.rect)
    Pts=[]
    while len(Pts)!=2:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()
            if event.type==pygame.MOUSEBUTTONDOWN:
                Pts.append(pygame.mouse.get_pos())
        if len(Pts)==2:
            break
        pygame.display.flip()
    dis1,dis2=inf,inf
    ans1,ans2=-1,-1
    for i in range(len(configPoints)):
        c=(configPoints[i][0],configPoints[i][1])
        d1=dist(Pts[0],c)
        d2=dist(Pts[1],c)
        if d1<dis1:
            dis1=d1
            ans1=i
        if d2<dis2:
            dis2=d2
            ans2=i
    pygame.quit()
    return ans1,ans2

MXAttempts=30
image=cv2.imread(image_name)
mapImage=image.copy()
NarrowPts=[]
corners=[(0,0),(image.shape[1],0),(image.shape[1],image.shape[0]),(0,image.shape[0])]
while MXAttempts:
    # generate a random point in an image whose pixel value is black
    while 1:
        Q1=(np.random.randint(0,Background.image.get_width()),np.random.randint(0,Background.image.get_height()),np.random.uniform(0,2*np.pi))
        # if grid value at Q1 is black break the loop
        if check_safe(Q1[2],Q1[0],Q1[1])==False:
            break
    thresh=10
    while 1:
        Q2=(Q1[0]+np.random.randint(-thresh,thresh),Q1[1]+np.random.randint(-thresh,thresh),Q1[2]+np.random.uniform(-np.pi/2,np.pi/2))
        if Euclidean(Q1,Q2)>thresh:
            continue
        # if grid value at Q1 is black break the loop
        if check_safe(Q2[2],Q2[0],Q2[1])==False:
            break
    
    # store mid point of Q1 and Q2 in Q
    Q=((Q1[0]+Q2[0])/2,(Q1[1]+Q2[1])/2,(Q1[2]+Q2[2])/2)
    if check_safe(Q[2],Q[0],Q[1])==False:
        continue
    flag=1
    for p in NarrowPts:
        if Euclidean(p,Q)<40:
            flag=0
            break
    if boundaryCheck(Q)==False:
        flag=0
    if flag==0:
        continue
    NarrowPts.append(Q)
    C1,C2,C3,C4=get_corners(Q[2],Q[0],Q[1])
    cv2.line(image,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),(0,0,255),2)
    # draw line from C2 to C3
    cv2.line(image,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),(0,0,255),2)
    # draw line from C1 to C2
    cv2.line(image,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),(0,0,255),2)
    # draw line from C3 to C4
    cv2.line(image,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),(0,0,255),2)
    cv2.imwrite('narrowOut.png',image)
    MXAttempts-=1
    

# print('Found Narrow Corridors')

image2=image.copy()
MXAttempts=20000
while MXAttempts:
    MXAttempts-=1
    Q=(np.random.randint(0,Background.image.get_width()),np.random.randint(0,Background.image.get_height()),np.random.uniform(0,2*np.pi))
    if check_safe(Q[2],Q[0],Q[1])==False:
        continue
    flag=0
    for p in NarrowPts:
        if Coli(p,Q):
            flag+=1

    if flag!=len(NarrowPts):
        continue
    # print('yo')
    NarrowPts.append(Q)
    C1,C2,C3,C4=get_corners(Q[2],Q[0],Q[1])
    cv2.line(image2,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),(255,0,0),2)
    # draw line from C2 to C3
    cv2.line(image2,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),(255,0,0),2)
    # draw line from C1 to C2
    cv2.line(image2,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),(255,0,0),2)
    # draw line from C3 to C4
    cv2.line(image2,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),(255,0,0),2)
    cv2.imwrite('narrow3Out.png',image2)


Ind={}
Par={}
def getPar(A):
    while Par[A]!=A:
        A=Par[A]
    return A
def uni(A,B):
    AA=getPar(A)
    BB=getPar(B)
    if AA!=BB:
        Par[AA]=BB

for i in range(len(NarrowPts)):
    Ind[NarrowPts[i]]=i 
    Par[i]=i

print('Done Q3')

image3=image2.copy()
MXAttempts=9000
NewPts=[]


while MXAttempts:
    MXAttempts-=1
    Q=(np.random.randint(0,Background.image.get_width()),np.random.randint(0,Background.image.get_height()),np.random.uniform(0,2*np.pi))
    if check_safe(Q[2],Q[0],Q[1])==False:
        continue
    flag=1
    lst=[]
    for p in NarrowPts:
        if Coli(p,Q)==False:
            lst.append(getPar(Ind[p]))
    # print(lst)
    lst=list(set(lst))
    if len(lst)==1:
        continue
    
    NarrowPts.append(Q)
    index=len(NarrowPts)-1
    Par[index]=index
    Ind[NarrowPts[index]]=index
    # lst.append(Q)
    for u in lst:
        for v in lst:
            uni(u,v)
    for p in NarrowPts:
        if lineCollision(p,Q)==False:
            uni(Ind[p],index)

    C1,C2,C3,C4=get_corners(Q[2],Q[0],Q[1])
    cv2.line(image3,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),(255,0,0),2)
    # draw line from C2 to C3
    cv2.line(image3,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),(255,0,0),2)
    # draw line from C1 to C2
    cv2.line(image3,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),(255,0,0),2)
    # draw line from C3 to C4
    cv2.line(image3,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),(255,0,0),2)
    cv2.imwrite('narrow4Out.png',image3)
print('Q4 Done')

image4=image3.copy()
n=100
for i in range(n):
    while 1:
        Q=(np.random.randint(0,Background.image.get_width()),np.random.randint(0,Background.image.get_height()),np.random.uniform(0,2*np.pi))
        if check_safe(Q[2],Q[0],Q[1]):
            C1,C2,C3,C4=get_corners(Q[2],Q[0],Q[1])
            cv2.line(image4,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),(255,255,0),2)
            # draw line from C2 to C3
            cv2.line(image4,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),(255,255,0),2)
            # draw line from C1 to C2
            cv2.line(image4,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),(255,255,0),2)
            # draw line from C3 to C4
            cv2.line(image4,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),(255,255,0),2)
            cv2.imwrite('Roadmap.png',image4)
            NewPts.append(Q)
            break

for pt in NewPts:
    configPoints.append(pt)
for pt in NarrowPts:
    configPoints.append(pt)

sourceindex,goalindex=getSourceGoal()
print(sourceindex,goalindex)
# sourceindex,goalindex=78,47
Source=configPoints[sourceindex]
Destination=configPoints[goalindex]
heap=[]
N=500
k=10
print(len(configPoints))
adjMatrix=[[0]*N for _ in range(N)]
for i in range(len(configPoints)):
    lst=[]
    for j in range(0,len(configPoints)):
        if i!=j:
            lst.append((Euclidean(configPoints[i],configPoints[j],50),j,configPoints[j]))

    lst.sort(key=lambda x:x[0])
    for j in range(0,k):
        if check_feasible(configPoints[i],lst[j][2]):
            adjMatrix[i][lst[j][1]]=adjMatrix[lst[j][1]][i]=1
            cv2.line(image3,(int(configPoints[i][0]),int(configPoints[i][1])),(int(lst[j][2][0]),int(lst[j][2][1])),(0,255,),1)

d=[inf]*N 
par=[-1]*N
print('Roadmap Initialized')
heapq.heappush(heap,(0,Source,sourceindex))
d[sourceindex]=0
MXITR=100000

while len(heap)>0 and MXITR:
    MXITR-=1
    (dist,node,j)=heapq.heappop(heap)
    if j==goalindex:
        print('Path found')
        break
    
    for i in range(0,len(configPoints)):
        if adjMatrix[j][i]:
            if dist+int(Euclidean(node,configPoints[i],50))<d[i]:
                d[i]=dist+int(Euclidean(node,configPoints[i],50))
                par[i]=j
                heapq.heappush(heap,(d[i],configPoints[i],i))

Path=[]
Cur=goalindex
while Cur!=-1:
    Path.append(configPoints[Cur])
    Cur=par[Cur]

Path.reverse()
image4=mapImage.copy()

# print(Path)

for i in range(len(Path)):
    col=(0,0,255)
    if i==0:
        col=(255,0,0)
    if i==len(Path)-1:
        col=(255,0,0)
    C1,C2,C3,C4=get_corners(Path[i][2],Path[i][0],Path[i][1])
    cv2.line(image4,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),col,2)
    # draw line from C2 to C3
    cv2.line(image4,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),col,2)
    # draw line from C1 to C2
    cv2.line(image4,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),col,2)
    # draw line from C3 to C4
    cv2.line(image4,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),col,2)
    if i>0:
        cv2.line(image4,(int(Path[i-1][0]),int(Path[i-1][1])),(int(Path[i][0]),int(Path[i][1])),(0,255,),2)


cv2.imwrite('narrow5.png',image4)
print('Done Q5')

image5=mapImage.copy()

# Part e begins
for i in range(0,len(Path)-1):
    contiPath=check_feasible(Path[i],Path[i+1],True,0.4)
    for j in range(0,len(contiPath)):
        col=(0,0,255)
        C1,C2,C3,C4=get_corners(contiPath[j][2],contiPath[j][0],contiPath[j][1])
        cv2.line(image5,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),col,2)
        cv2.line(image5,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),col,2)
        cv2.line(image5,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),col,2)
        cv2.line(image5,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),col,2)
        

cv2.imwrite('Narrow6.png',image5)
