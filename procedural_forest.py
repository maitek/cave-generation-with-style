import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

map_size = (100,100)
num_trees = 500
random_pos = np.random.rand(num_trees,2)
#import pdb; pdb.set_trace()
#plt.plot(random_pos,".g")
#plt.show()

class DistanceHashMap():
    def __init__(self, grid_size):
        self.hashmap = dict()
        self.grid_size = grid_size

    def insert(self,point):
        """ Add (x,y) into data structure, O(1)"""
        x = point[0]
        y = point[1]
        xyhash = str(int(round(x/self.grid_size))) + str(int(round(y/self.grid_size)))

        if self.hashmap.get(xyhash) is None:
            self.hashmap[xyhash] = list()
        self.hashmap[xyhash].append(np.asarray((x,y)))
        #import pdb; pdb.set_trace()

    def get_hash_points(self,point):
        """ Get all points in hash bin containing (x,y), O(1)"""
        x = point[0]
        y = point[1]
        xyhash = str(int(round(x/self.grid_size))) + str(int(round(y/self.grid_size)))
        if self.hashmap.get(xyhash) is not None:
            return self.hashmap[xyhash]
        else:
            return []

    def get_all_points(self):
        points = list()
        for key in self.hashmap.keys():
            points += self.hashmap[key]
        return points

    def get_nearest_points(self,point, distance = 1):
        """ Get all points near points to (x,y)
            Garanties all points within distance are included
        """
        x = point[0]
        y = point[1]
        points = list()

        xmin = int(round(x-distance))
        xmax = int(round(x+distance))+1
        ymin = int(round(y-distance))
        ymax = int(round(y+distance))+1

        # combine all points from neighboring hashbin
        for xi in range(xmin,xmax):
            for yi in range(ymin,ymax):
                points += self.get_hash_points((xi,yi))
        return points

def check_valid(new_point,points):
    for point in points:
        if norm(new_point-point) < 0.8:
            return False
    return True

points = list()
edge_points = list()
num_points = 1

# create first point at orgin
new_point = np.asarray((0,0))
points_map = DistanceHashMap(grid_size=1)
points_map.insert(new_point)

edge_points.append(new_point)

while num_points < 10000 and len(edge_points) > 0:
    # select random edge
    idx = np.random.randint(0,len(edge_points))
    seed_point = edge_points[idx]

    # create new point with a fixed distance and random angle from the edge
    dist = 1

    valid = False
    attempts = list()
    for i in range(5):

        angle_rad = np.random.rand(1)*np.pi*2
        x = np.sin(angle_rad)*dist
        y = np.cos(angle_rad)*dist
        new_point = seed_point + np.asarray((x[0],y[0]))
        #import pdb; pdb.set_trace()
        attempts.append(new_point)
        points = points_map.get_nearest_points(new_point)
        valid = check_valid(new_point,points)
        if valid:
            break

    if not valid:
        edge_points.pop(idx)
    else:
        points_map.insert(new_point)
        edge_points.append(new_point.copy())
        num_points+=1
        #print(num_points)
        print(len(points))

        if num_points % 500 == 0:
            #import pdb; pdb.set_trace()

            points_array = np.asarray(points_map.get_all_points())
            edge_points_array = np.asarray(edge_points)
            attempts_array = np.asarray(attempts)
            plt.plot(points_array[:,0],points_array[:,1],".g")
            plt.plot(edge_points_array[:,0],edge_points_array[:,1],".y")
            plt.plot(seed_point[0],seed_point[1],".r")
            plt.plot(attempts_array[:,0],attempts_array[:,1],".b")
            plt.show()



# while(num_points < max_points):
    # add point at current_pos, and mark it edge
    # num_points += 1
    # select a random edge point
        # if is_still_edge():
            # current_pos = create_new_pos(current_pos)
        # else:
            # mark as not edge
