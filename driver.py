from RRT_Star import *
from utils import *

# Define the robot type and other parameters
start = (187, 341)
robot_radius=20
robot=MobileRobot((start[0],start[1]),robot_radius)

goal = (382, 72)

# RRT* specific parameters
step_size = 20
max_iterations = 1000

occupancy_map = grid


# Create an instance of the RRTStar class
driverRRT = RRTStar(robot_type=robot, start=start, goal=goal, obstacles_list=Polygons, step_size=step_size,radius_obstacle=radius_obstacle)
assert(driverRRT.collision_free(robot.pose))

markedImage=mapImage.copy()
cv2.circle(markedImage,(int(start[0]),int(start[1])),radius_obstacle,(255,0,0),-1)
cv2.circle(markedImage,(int(goal[0]),int(goal[1])),radius_obstacle,(0,0,255),-1)
treeImage=markedImage.copy()

# Run the RRT* algorithm
while max_iterations:
    eps=np.random.uniform()
    q = driverRRT.get_random_config(Background.image.get_width(),Background.image.get_height())
    if eps<0.2:
        q=goal
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
    
cv2.imwrite("path.png",pathImage)
print(path)