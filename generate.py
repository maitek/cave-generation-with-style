import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

import tensorflow as tf
import os

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops

# Map options
map_size = (256,256)
output_size = (512,512)
num_iterations = 10
ksize = 7
random_seed = 3

# Style options
model_checkpoint = "multistyle-pastiche-generator-varied.ckpt"
model_checkpoint = os.path.join(os.getcwd(),model_checkpoint)

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

def load_checkpoint(sess, checkpoint):
  """Loads a checkpoint file into the session."""
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)

def stylize_image(input_image, which_styles, checkpoint):

    # convert image to tensor
    tensor = input_image[...,np.newaxis].astype(np.float32)
    tensor = np.dstack((tensor,tensor,tensor))
    tensor = tensor[np.newaxis,...]

    """Stylizes an image into a set of styles and writes them to disk."""
    with tf.Graph().as_default(), tf.Session() as sess:
        style_network = model.transform(
            tf.concat([tensor for _ in range(len(which_styles))], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': 32,
                'center': True,
                'scale': True})
        load_checkpoint(sess, checkpoint)

        output_image = style_network.eval()
        return output_image[0,:,:,:]


def generate_noise_map(map_size):
    for i in range(1,100):
    	noise_map = np.zeros(map_size)
    	for octave in [1,2,8,32]:
    		noise_map_octave = np.random.rand(int(map_size[0]/octave),int(map_size[1]/octave))
    		#print(noise_map_octave)
    		noise_map_octave = imresize(noise_map_octave ,map_size, interp="bicubic", mode='F')
    		noise_map += noise_map_octave*octave

    return noise_map


# Generate with all styles
for i in range(32):
    print("Generating map in style {}".format(i))
    # generate map and resize it to output size
    random_map = generate_map(map_size,num_iterations,ksize)
    random_map = imresize(random_map,output_size, interp="nearest")
    input_map = random_map.astype(np.float32) + generate_noise_map(random_map.shape)*10

    map_with_style = stylize_image(input_map,[i],model_checkpoint)
    imsave('style_{}.png'.format(i), map_with_style)
    imsave('map_{}.png'.format(i), random_map)

# show map
plt.imshow(random_map)
plt.imshow(map_with_style)
plt.show()
