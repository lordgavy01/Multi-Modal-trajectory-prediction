from RRT_Star import *
from utils import *

# Define the robot type and other parameters
start = (219, 334)
robot_radius=20
robot=MobileRobot((start[0],start[1]),robot_radius)

goal = (428, 72)

# RRT* specific parameters
step_size = 10
max_iterations = 1000

occupancy_map = grid


# Create an instance of the RRTStar class
driverRRT = RRTStar(robot_type=robot, start=start, goal=goal, obstacles_list=Polygons, step_size=step_size,radius_obstacle=radius_obstacle)
assert(driverRRT.collision_free(robot.pose))

markedImage=mapImage.copy()
cv2.circle(markedImage,(int(start[0]),int(start[1])),radius_obstacle,(255,0,0),-1)
cv2.circle(markedImage,(int(goal[0]),int(goal[1])),radius_obstacle,(0,0,255),-1)

Checkpoint0=(436,238)
# Checkpoint1=(64, 126)
# Checkpoint2=(251, 146)
# cv2.circle(markedImage,(int(Checkpoint0[0]),int(Checkpoint0[1])),radius_obstacle,(255,0,0),-1)
# cv2.circle(markedImage,(int(Checkpoint1[0]),int(Checkpoint1[1])),radius_obstacle,(255,0,0),-1)
# cv2.circle(markedImage,(int(Checkpoint2[0]),int(Checkpoint2[1])),radius_obstacle,(0,0,255),-1)

# for obs in driverRRT.obstacles:
#     cv2.circle(markedImage,(int(obs[0]),int(obs[1])),radius_obstacle,(0,0,0),-1)
treeImage=markedImage.copy()

driverRRT.goal=Checkpoint0
# Run the RRT* algorithm
while max_iterations:
    
    eps=np.random.uniform()
    q = driverRRT.get_random_config(Background.image.get_width(),Background.image.get_height())
    if eps<0.6:
        q=driverRRT.goal 
    
    if max_iterations==965:
        driverRRT.obstacles=driverRRT.obstacles[:3]
        driverRRT.goal=goal
        for obs in driverRRT.obstacles:
            cv2.circle(treeImage,(int(obs[0]),int(obs[1])),radius_obstacle,(255,255,255),-1)

      

    q_near_index = driverRRT.nearest_vertex(q)
    q_near=driverRRT.vertices[q_near_index]
    q_new = driverRRT.steer(q_near, q)
    bufferImage=markedImage.copy()
    # cv2.circle(bufferImage,(int(q_new[0]),int(q_new[1])),robot_radius,(255,255,0),-1)
    # cv2.imwrite('buf.png',bufferImage)
    # input()
    if q_new[0]<0 or q_new[0]>Background.image.get_width() or q_new[1]<0 or q_new[1]>Background.image.get_height():
        # print('Not Passed')
        continue
    if not driverRRT.collision_free(q_new):
        # print('Not Passed')
        continue
    if q_new in driverRRT.vertices:
        # print('Not Passed')
        continue
    if driverRRT.collision_free_path(q_near, q_new):
        print(max_iterations)
        max_iterations-=1
        driverRRT.add_vertex(q_new,q_near_index)
        driverRRT.rewire(q_new)
        cv2.circle(treeImage,(int(q_new[0]),int(q_new[1])),5,(0,0,255),-1)
        cv2.imwrite("tree.png",treeImage)
        # Check if we've reached the goal
        if driverRRT.goal==goal and driverRRT.distance(q_new, driverRRT.goal) < driverRRT.goal_tolerance:
            print('Goal Reached')
            driverRRT.add_vertex(goal,q_near_index)
            driverRRT.rewire(goal)
            cv2.imwrite("tree.png",treeImage)
            break

pathImage=markedImage.copy()
# Extract the path from the RRT*
path = driverRRT.get_path_to_goal()
path.reverse()
for points in path:
    cv2.circle(pathImage,(int(points[0]),int(points[1])),robot_radius,(0,255,0))

with open("paths.txt","a") as f:
    f.write('Corrected Path: ')
    for points in path:
        f.write('('+str(points[0])+','+str(points[1])+') ')
    f.write('\n')
cv2.imwrite("path.png",pathImage)
print(path)