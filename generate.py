import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imsave, imresize

import tensorflow as tf

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model
from magenta.models.image_stylization import ops

# Map options
map_size = (256,256)
output_size = (1024,1024)
num_iterations = 10
ksize = 7
random_seed = 3

# Style options
model_checkpoint = "multistyle-pastiche-generator-varied.ckpt"


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

def stylize_image(image,checkpoint):
    #def _multiple_styles(input_image, which_styles, output_dir):
    #  """Stylizes image into a linear combination of styles and writes to disk."""
    X = image[np.newaxis,...,np.newaxis].astype(np.float32)
    num_styles = 32
    weights = np.zeros([num_styles], dtype=np.float32)
    weights[0] = 1

    print(X.shape, X.dtype)

    with tf.Graph().as_default(), tf.Session() as sess:
        style_network = model.transform(
            X,
            normalizer_fn=ops.weighted_instance_norm,
            normalizer_params={
                'weights': tf.constant(weights),
                'num_categories': num_styles,
                'center': True,
                'scale': True})
        load_checkpoint(sess, checkpoint)

        stylized_image = style_network.eval()
    return stylize_image


# generate map and resize it to output size
random_map = generate_map(map_size,num_iterations,ksize)
random_map = imresize(random_map,output_size, interp="nearest")

#map_with_style = stylize_image(random_map,model_checkpoint)

#save map
imsave('map.png', random_map)
#imsave('map_with_style.png', map_with_style)
# show map
plt.imshow(random_map)
#plt.imshow(map_with_style)
plt.show()
