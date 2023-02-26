from RRT_Star import *
from utils import *

# Define the robot type and other parameters

robot_base=(246,300)

robot=Manipulator(robot_base)
start = (381, 85)
goal = (91, 101)

# RRT* specific parameters
step_size = 10
max_iterations = 1000

occupancy_map = grid


# Create an instance of the RRTStar class
driverRRT = RRTStar(robot_type=robot, start=start, goal=goal, obstacles_list=Polygons, step_size=step_size,radius_obstacle=radius_obstacle)
start=driverRRT.get_approx_config(start)
goal=driverRRT.get_approx_config(goal)
robot.pose=driverRRT.start=start
driverRRT.goal=goal
assert(driverRRT.collision_free(robot.pose))

markedImage=mapImage.copy()
treeImage=markedImage.copy()

# Run the RRT* algorithm
while max_iterations:
    eps=np.random.uniform()
    q = driverRRT.get_random_config(Background.image.get_width(),Background.image.get_height())
    
    q_near_index = driverRRT.nearest_vertex(q)
    q_near=driverRRT.vertices[q_near_index]
    q_new = driverRRT.steer(q_near, q)
    if q_new[0]<0 or q_new[0]>Background.image.get_width() or q_new[1]<0 or q_new[1]>Background.image.get_height():
        continue
    if not driverRRT.collision_free(q_new):
        continue
    if driverRRT.collision_free_path(q_near, q_new):
        print(max_iterations)
        max_iterations-=1
        driverRRT.add_vertex(q_new,q_near_index)
        driverRRT.rewire(q_new)
        cv2.circle(treeImage,(int(q_new[0]),int(q_new[1])),5,(0,0,255),-1)
        cv2.imwrite("tree.png",treeImage)
        # Check if we've reached the goal
        if driverRRT.distance(q_new, goal) < driverRRT.goal_tolerance:
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
    cv2.circle(pathImage,(int(points[0]),int(points[1])),radius_obstacle,(0,255,0))

with open("paths.txt","a") as f:
    f.write('Original Path: ')
    for points in path:
        f.write('('+str(points[0])+','+str(points[1])+') ')
    f.write('\n')
cv2.imwrite("path.png",pathImage)
print(path)