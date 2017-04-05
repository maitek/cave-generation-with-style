import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imsave, imresize

# Map options
map_size = (256,256)
output_size = (1024,1024)
num_iterations = 10
ksize = 7
random_seed = 3

def generate_map(map_size=(64,64),num_iterations=10, ksize = 3, random_seed=None):
    """
        Generate a random map using cellular automata
    """
    # generate random binary image
    random_map = np.random.rand(map_size[0],map_size[1])
    random_map[random_map < 0.5] = 0
    random_map[random_map > 0.5] = 1

    # cellular automata
    for k in range(num_iterations):
        kernel = np.ones((ksize,ksize))
        conv_map = convolve2d(random_map,kernel,mode='same')
        threshold = (ksize*ksize)/2
        random_map[conv_map <= threshold] = 0 # 4 rule
        random_map[conv_map > threshold] = 1 # 5 rule
    return random_map


# generate map and resize it to output size
random_map = generate_map(map_size,num_iterations,ksize)
random_map = imresize(random_map,output_size, interp="nearest")

#save map
imsave('map.png', random_map)

# show map
plt.imshow(random_map)
plt.show()
