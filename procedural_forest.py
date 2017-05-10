import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

map_size = (100,100)
num_trees = 500
random_pos = np.random.rand(num_trees,2)
#import pdb; pdb.set_trace()
#plt.plot(random_pos,".g")
#plt.show()



def check_valid(new_point,points):
    for point in points:
        if norm(new_point-point) < 0.8:
            return False
    return True

points = list()
edge_points = list()
num_points = 1

# create first point at orgin
new_point = np.zeros((1,2))
points.append(new_point)
edge_points.append(new_point)

while num_points < 1000 and len(edge_points) > 0:
    # select random edge
    idx = np.random.randint(0,len(edge_points))
    seed_point = edge_points[idx]


    # create new point with a fixed distance and random angle from the edge

    dist = 1

    valid = False
    attempts = list()
    for i in range(5):
        angle_rad = np.random.rand(1)*np.pi*2
        new_point = seed_point + np.asarray([np.sin(angle_rad)*dist,np.cos(angle_rad)*dist]).T
        attempts.append(new_point)
        valid = check_valid(new_point,points)
        if valid:
            print("new",new_point)
            break

    if not valid:
        edge_points.pop(idx)
        print("removed",idx)
    else:
        points.append(new_point.copy())
        edge_points.append(new_point.copy())
        num_points+=1
        print(num_points)

        if num_points % 100 == 0:
            import pdb; pdb.set_trace()
            print(points)
            points_array = np.asarray(points)[:,0,:]
            edge_points_array = np.asarray(edge_points)[:,0,:]
            attempts_array = np.asarray(attempts)[:,0,:]
            seed_point_array = np.asarray(seed_point)[0]
            plt.plot(points_array[:,0],points_array[:,1],".g")
            plt.plot(edge_points_array[:,0],edge_points_array[:,1],".y")
            #plt.plot(seed_point_array[0],seed_point_array[1],".r")
            #plt.plot(attempts_array[:,0],attempts_array[:,1],".b")
            plt.show()



# while(num_points < max_points):
    # add point at current_pos, and mark it edge
    # num_points += 1
    # select a random edge point
        # if is_still_edge():
            # current_pos = create_new_pos(current_pos)
        # else:
            # mark as not edge
