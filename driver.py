from RRT_Star import *
from utils import *

# Define the robot type and other parameters
start = (68, 403)
robot=MobileRobot((start[0],start[1],pi/2),45)
assert(robot.check_safe(robot.pose))
goal = (432, 95, pi/3)

# RRT* specific parameters
rewiring_radius = 1.0
step_size = 0.2
max_iterations = 10

occupancy_map = grid

# Create an instance of the RRTStar class
driverRRT = RRTStar(robot_type=robot, start=start, goal=goal, occupancy_map=occupancy_map, rewiring_radius=rewiring_radius, step_size=step_size)

# Run the RRT* algorithm
for i in range(max_iterations):
    q = driverRRT.get_random_config()
    q_near = driverRRT.nearest_vertex(q)
    q_new = driverRRT.steer(q_near, q)
    if driverRRT.collision_free_path(q_near, q_new):
        nearby_vertices = rrt.nearby_vertices(q_new)
        rrt.add_vertex(q_new)
        rrt.add_edges_to_neighbors(q_new, nearby_vertices)
        rrt.rewire(q_new)

        # Check if we've reached the goal
        if rrt.distance(q_new, goal) < step_size:
            rrt.add_vertex(goal)
            rrt.add_edges_to_neighbors(goal, rrt.nearby_vertices(goal))
            rrt.rewire(goal)
            break

# Extract the path from the RRT*
path = rrt.get_path_to_goal()

# Visualize the path and the RRT*
...  # some code for visualization
