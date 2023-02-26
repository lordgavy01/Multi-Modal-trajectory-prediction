import numpy as np
from math import *
import pygame
import cv2
import heapq


np.random.seed(22)
image_name='new_geomap.png'
pygame.init()
class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

Background=Background(image_name,[0,0])

def get_corners(theta,X,Y,L,B): 
    l=L/2
    b=B/2
    C1=(X+l*cos(theta)+b*cos(pi/2+theta),Y+l*sin(theta)+b*sin(pi/2+theta))
    C2=(X-l*cos(theta)+b*cos(pi/2+theta),Y-l*sin(theta)+b*sin(pi/2+theta))
    C3=(X-l*cos(theta)-b*cos(pi/2+theta),Y-l*sin(theta)-b*sin(pi/2+theta))
    C4=(X+l*cos(theta)-b*cos(pi/2+theta),Y+l*sin(theta)-b*sin(pi/2+theta))
    return C1,C2,C3,C4

def SEuclidean(A,B):
    ans=0
    for i in range(len(A)):
        ans+=pow(A[i]-B[i],2)
    return sqrt(ans)

grid=[[1]*Background.image.get_width() for _ in range(Background.image.get_height())]

for i in range(Background.image.get_height()):
    for j in range(Background.image.get_width()):
        if Background.image.get_at((j,i))==(0,0,0,255):
            grid[i][j]=0

Polygons=[]
# Read list of Polygons from out.txt where one line in that file is list of points of one polygon
with open('new_out.txt') as f:
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


num_Links=3
lengths=[]
for i in range(num_Links):
    lengths.append((np.random.randint(60,80),10))

Base=(355, 320)
configPoints=[]
image=cv2.imread(image_name)
base=image.copy()

cv2.circle(base,Base,10,(0,0,255),-1)
cv2.imwrite('base.jpg',base)

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
    Poly1=Pol1[1]
    Poly2=Pol2[1]
    x1=min(Poly1,key=lambda x:x[0])[0]
    x2=max(Poly1,key=lambda x:x[0])[0]
    y1=min(Poly1,key=lambda x:x[1])[1]
    y2=max(Poly1,key=lambda x:x[1])[1]
    x3=min(Poly2,key=lambda x:x[0])[0]
    x4=max(Poly2,key=lambda x:x[0])[0]
    y3=min(Poly2,key=lambda x:x[1])[1]
    y4=max(Poly2,key=lambda x:x[1])[1]
    if x1>x4 or x2<x3 or y1>y4 or y2<y3:
        return True
    return False

def getEndPoint(thetas):
    Lines=[]
    P1=(Base[0]+lengths[0][0]*cos(thetas[0]),Base[1]+lengths[0][0]*sin(thetas[0]))
    old_theta=0
    old_point=P1
    Lines.append((Base,P1))
    for i in range(1,len(thetas)):
        old_theta+=thetas[i-1]
        new_theta=thetas[i]
        cur_point=(old_point[0]+lengths[i][0]*cos(new_theta+old_theta),old_point[1]+lengths[i][0]*sin(new_theta+old_theta))
        Lines.append((old_point,cur_point))
        old_point=cur_point
    return Lines[-1][1]
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
        c=getEndPoint(configPoints[i])
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

def check_safe(thetas,draw=False,image=None,col=(0,0,255)):
    Poly=[]
    Lines=[]
    # print(thetas)
    P1=(Base[0]+lengths[0][0]*cos(thetas[0]),Base[1]+lengths[0][0]*sin(thetas[0]))
    # Make a variable mid that is midpoint of Base and P1
    Mid=((Base[0]+P1[0])/2,(Base[1]+P1[1])/2)
    C1,C2,C3,C4=get_corners(thetas[0],Mid[0],Mid[1],lengths[0][0],lengths[0][1])
    Poly.append([Mid,[C1,C4,C3,C2]])
    old_theta=0
    old_point=P1
    Lines.append((Base,P1))
    for i in range(1,len(thetas)):
        old_theta+=thetas[i-1]
        new_theta=thetas[i]
        cur_point=(old_point[0]+lengths[i][0]*cos(new_theta+old_theta),old_point[1]+lengths[i][0]*sin(new_theta+old_theta))
        Mid=((old_point[0]+cur_point[0])/2,(old_point[1]+cur_point[1])/2)
        C1,C2,C3,C4=get_corners(new_theta+old_theta,Mid[0],Mid[1],lengths[i][0],lengths[i][1])
        Poly.append([Mid,[C1,C4,C3,C2]])
        Lines.append((old_point,cur_point))
        old_point=cur_point

    for pol in Poly:
        for obs_pol in Polygons:
            if dont_intersect(pol,obs_pol):
                continue
            if check_intersect(pol,obs_pol)!='No':
                return False
        for c in pol[1]:
            if c[0]>=0 and c[1]>=0 and c[0]<Background.image.get_width() and c[1]<Background.image.get_height():
                if grid[int(c[1])][int(c[0])]==0:
                    return False
            else:
                return False
    for i in range(len(Poly)):
        for j in range(i+2,len(Poly)):
            if check_intersect(Poly[i],Poly[j])!='No':
                # print('Self Collision')
                return False

    if draw:
        for pol in Poly:
            # cv2.circle(image,(int(pol[0][0]),int(pol[0][1])),5,(0,0,255),-1)
            # pygame.draw.circle(Background.image,(255,0,0),pol[0],5)
            C1,C4,C3,C2=pol[1]
            cv2.line(image,(int(C1[0]),int(C1[1])),(int(C4[0]),int(C4[1])),col,2)
            # draw line from C2 to C3
            cv2.line(image,(int(C2[0]),int(C2[1])),(int(C3[0]),int(C3[1])),col,2)
            # draw line from C1 to C2
            cv2.line(image,(int(C1[0]),int(C1[1])),(int(C2[0]),int(C2[1])),col,2)
            # draw line from C3 to C4
            cv2.line(image,(int(C3[0]),int(C3[1])),(int(C4[0]),int(C4[1])),col,2)
    return True

def check_feasible(thetas1,thetas2,draw=False,image=None,EPS=None):
    Poly1=[]
    P1=(Base[0]+lengths[0][0]*cos(thetas1[0]),Base[1]+lengths[0][0]*sin(thetas1[0]))
    Mid=((Base[0]+P1[0])/2,(Base[1]+P1[1])/2)
    C1,C2,C3,C4=get_corners(thetas1[0],Mid[0],Mid[1],lengths[0][0],lengths[0][1])
    Poly1.append([Mid,[C1,C4,C3,C2]])
    old_theta=0
    old_point=P1
    for i in range(1,len(thetas1)):
        old_theta+=thetas1[i-1]
        new_theta=thetas1[i]
        cur_point=(old_point[0]+lengths[i][0]*cos(new_theta+old_theta),old_point[1]+lengths[i][0]*sin(new_theta+old_theta))
        Mid=((old_point[0]+cur_point[0])/2,(old_point[1]+cur_point[1])/2)
        C1,C2,C3,C4=get_corners(new_theta+old_theta,Mid[0],Mid[1],lengths[i][0],lengths[i][1])
        Poly1.append([Mid,[C1,C4,C3,C2]])
        old_point=cur_point
    Poly2=[]
    P1=(Base[0]+lengths[0][0]*cos(thetas2[0]),Base[1]+lengths[0][0]*sin(thetas2[0]))
    Mid=((Base[0]+P1[0])/2,(Base[1]+P1[1])/2)
    C1,C2,C3,C4=get_corners(thetas2[0],Mid[0],Mid[1],lengths[0][0],lengths[0][1])
    Poly2.append([Mid,[C1,C4,C3,C2]])
    old_theta=0
    old_point=P1
    for i in range(1,len(thetas2)):
        old_theta+=thetas2[i-1]
        new_theta=thetas2[i]
        cur_point=(old_point[0]+lengths[i][0]*cos(new_theta+old_theta),old_point[1]+lengths[i][0]*sin(new_theta+old_theta))
        Mid=((old_point[0]+cur_point[0])/2,(old_point[1]+cur_point[1])/2)
        C1,C2,C3,C4=get_corners(new_theta+old_theta,Mid[0],Mid[1],lengths[i][0],lengths[i][1])
        Poly2.append([Mid,[C1,C4,C3,C2]])
        old_point=cur_point 
    
    for i in range(len(Poly1)):
        M1,_=Poly1[i]
        M2,_=Poly2[i]
        if M1==M2:
            print('M1',M1)
            print('M2',M2)
            print(Poly1[i][1])
            print(Poly2[i][1])
            print(thetas1)
            print(thetas2)
            print(Poly1)
            print(Poly2)
        
        L=0
        if EPS==None:
            eps=1/SEuclidean(M1,M2)
        else:
            eps=EPS
        # print(eps)
        while L<=1:
            C=(M1[0]*(1-L)+M2[0]*L,M1[1]*(1-L)+M2[1]*L)            
            C1,C2,C3,C4=get_corners(atan2(C2[1]-C1[1],C2[0]-C1[0]),C[0],C[1],lengths[i][0],lengths[i][1])
            pol=[C,[C1,C4,C3,C2]]
            for obs_pol in Polygons:
                if dont_intersect(pol,obs_pol):
                    continue
                if check_intersect(pol,obs_pol)!='No':
                    return False
            L+=eps
    if draw:
        End1=Poly1[-1][0]
        End2=Poly2[-1][0]
        cv2.line(image,(int(End1[0]),int(End1[1])),(int(End2[0]),int(End2[1])),(255,0,0),1)
    return True

def getDis(G,L):
    P=(G[0],G[1])
    # return shortest distance between point P and line segment L which joins L[0] and L[1]
    A=P[0]-L[0][0]
    B=P[1]-L[0][1]
    C=L[1][0]-L[0][0]
    D=L[1][1]-L[0][1]
    dot=A*C+B*D
    len_sq=C*C+D*D
    param=dot/len_sq
    if param<0:
        return dist(P,L[0])
    elif param>1:
        return dist(P,L[1])
    else:
        return dist(P,(L[0][0]+param*C,L[0][1]+param*D))

def helpClearance(Q):
    # return the clearance of point Q
    mx=inf
    for pol in Polygons:
        for i in range(len(pol[1])):
            Line=[pol[1][i],pol[1][(i+1)%len(pol[1])]]
            dis=getDis(Q,Line)
            mx=min(mx,dis)
    return mx

def getClearance(thetas):
    Poly=[]
    Lines=[]
    P1=(Base[0]+lengths[0][0]*cos(thetas[0]),Base[1]+lengths[0][0]*sin(thetas[0]))
    # Make a variable mid that is midpoint of Base and P1
    Mid=((Base[0]+P1[0])/2,(Base[1]+P1[1])/2)
    C1,C2,C3,C4=get_corners(thetas[0],Mid[0],Mid[1],lengths[0][0],lengths[0][1])
    Poly.append([Mid,[C1,C4,C3,C2]])
    old_theta=0
    old_point=P1
    Lines.append((Base,P1))
    for i in range(1,len(thetas)):
        old_theta+=thetas[i-1]
        new_theta=thetas[i]
        cur_point=(old_point[0]+lengths[i][0]*cos(new_theta+old_theta),old_point[1]+lengths[i][0]*sin(new_theta+old_theta))
        Mid=((old_point[0]+cur_point[0])/2,(old_point[1]+cur_point[1])/2)
        C1,C2,C3,C4=get_corners(new_theta+old_theta,Mid[0],Mid[1],lengths[i][0],lengths[i][1])
        Poly.append([Mid,[C1,C4,C3,C2]])
        Lines.append((old_point,cur_point))
        old_point=cur_point

    mx=inf
    for i in range(len(Poly)):
        for j in range(len(Poly[i][1])):
            A=Poly[i][1][j]
            B=Poly[i][1][(j+1)%4]
            L=0
            eps=1/dist(A,B)
            while L<=1:
                C=(A[0]*(1-L)+B[0]*L,A[1]*(1-L)+B[1]*L)
                mx=min(mx,helpClearance(C))
                L+=eps
    return mx

Thresh=5 # threshold for min dist between samples makei it>=6.3 if want far away samples.
FarBool=True
MXAttempts=4000
# image=cv2.imread(image_name)
image=base.copy()
mapImage=base.copy()
NarrowPts=[]
EndPts=[]
corners=[(0,0),(image.shape[1],0),(image.shape[1],image.shape[0]),(0,image.shape[0])]
while MXAttempts:
    MXAttempts-=1
    # generate a random point in an image whose pixel value is black
    while 1:
        Q1=[]
        for j in range(num_Links):
            theta=np.random.uniform(0,2*pi)
            Q1.append(theta)
        if check_safe(Q1)==False:
            break
    thresh=pi/10
    while 1:
        Q2=[0]*num_Links
        for j in range(num_Links):
            Q2[j]=Q1[j]+np.random.uniform(-thresh,thresh)
        if check_safe(Q2)==False:
            break
    
    Q=[0]*num_Links
    for j in range(num_Links):
        Q[j]=(Q1[j]+Q2[j])/2
    
    flag=1
    curEndPt=getEndPoint(Q)
    if FarBool:
        for p in NarrowPts:
            if SEuclidean(p,Q)<=Thresh:
                flag=0
                break
    if flag==0:
        continue
    if check_safe(Q,True,image=image)==False:
        continue

    EndPts.append(curEndPt)
    NarrowPts.append(Q)
    cv2.imwrite('narrowOut.png',image)
    
print('Found Narrow Corridors')

image2=image.copy()
MXAttempts=2000
while MXAttempts:
    MXAttempts-=1
    Q=[]
    for j in range(num_Links):
        theta=np.random.uniform(0,2*pi)
        Q.append(theta)
    if not check_safe(Q):
        continue
    if FarBool:
        for pt in NarrowPts:
            if SEuclidean(pt,Q)<=Thresh:
                continue
    val=getClearance(Q)
    if val<14:
        continue
    check_safe(Q,True,image=image2,col=(255,0,0))
    NarrowPts.append(Q)
    cv2.imwrite('ClearanceOut.png',image2)

NewPts=[]
image4=image2.copy()
n=60
for i in range(n):
    while 1:
        Q=[]
        for j in range(num_Links):
            theta=np.random.uniform(0,2*pi)
            Q.append(theta)
        if check_safe(Q,True,image=image4,col=(255,255,0)):
            cv2.imwrite('Roadmap.png',image4)
            NewPts.append(Q)
            break

for pt in NewPts:
    configPoints.append(pt)
for pt in NarrowPts:
    configPoints.append(pt)

sourceindex,goalindex=getSourceGoal()
print(sourceindex,goalindex)
# sourceindex,goalindex=40 71

Source=configPoints[sourceindex]
Destination=configPoints[goalindex]
heap=[]
heap=[]
N=1000
k=10
print(len(configPoints))
adjMatrix=[[0]*N for _ in range(N)]
for i in range(len(configPoints)):
    lst=[]
    for j in range(0,len(configPoints)):
        if i!=j:
            lst.append((dist(configPoints[i],configPoints[j]),j,configPoints[j]))

    lst.sort(key=lambda x:x[0])
    for j in range(0,k):
        if check_feasible(configPoints[i],lst[j][2]):
            try:
                adjMatrix[i][lst[j][1]]=adjMatrix[lst[j][1]][i]=1
            except:
                print(len(configPoints),i,j,len(adjMatrix[i]),len(adjMatrix[j]))
                exit(0)
            # cv2.line(image3,(int(configPoints[i][0]),int(configPoints[i][1])),(int(lst[j][2][0]),int(lst[j][2][1])),(0,255,),1)

d=[inf]*N 
par=[-1]*N
print('Roadmap Initialized')
heapq.heappush(heap,(0,Source,sourceindex))
d[sourceindex]=0
MXITR=100000

while len(heap)>0 and MXITR:
    MXITR-=1
    (dis,node,j)=heapq.heappop(heap)
    if j==goalindex:
        print('Path found')
        break
    for i in range(0,len(configPoints)):
            if adjMatrix[j][i]:
                if dis+int(100*SEuclidean(node,configPoints[i]))<d[i]:
                    d[i]=dis+int(100*SEuclidean(node,configPoints[i]))
                    par[i]=j
                    heapq.heappush(heap,(d[i],configPoints[i],i))


Path=[]
Cur=goalindex
while Cur!=-1:
    Path.append(configPoints[Cur])
    Cur=par[Cur]

Path.reverse()
image4=mapImage.copy()

for i in range(len(Path)):
    col=(0,0,255)
    if i==0:
        col=(255,0,0)
    if i==len(Path)-1:
        col=(255,0,0)
    thetas=Path[i].copy()
    img=base.copy()
    check_safe(thetas,True,img,col)
    cv2.imwrite('D/'+str(i)+'.png',img)
cv2.imwrite('narrow5.png',image4)

print('Done Q5')
