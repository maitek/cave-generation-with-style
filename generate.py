import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

import tensorflow as tf
import os

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model


def generate_map(map_size=(64,64),num_iterations=10, ksize = 3):
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

def main():

    # Map options
    map_size = (256,256)
    output_size = (1024,1024)
    num_iterations = 10
    ksize = 9

    # TODO download model automatically

    # Style options
    model_checkpoint = "multistyle-pastiche-generator-varied.ckpt"
    model_checkpoint = os.path.join(os.getcwd(),model_checkpoint)

    # Generate caves with all styles
    for i in range(32):

        print("Generating map in style {}".format(i))

        # generate map and resize it to output size
        random_map = generate_map(map_size,num_iterations,ksize)
        random_map = imresize(random_map,output_size, interp="nearest")
        random_map = random_map.astype(np.float32)/255
        mask = np.dstack((random_map,random_map,random_map)).astype(np.float32)

        # generate a random texture to give the level some more detail
        noise_map = np.random.rand(random_map.shape[0],random_map.shape[1])
        bg_style_map = stylize_image(noise_map,[np.random.randint(32)],model_checkpoint)
        bg_map =  bg_style_map * mask
        fg_map = np.swapaxes(bg_style_map,0,1) * (np.ones_like(mask)-mask)
        combined_map = bg_map+fg_map*0.1

        # apply style transfer to get stylized levels
        map_style = stylize_image(np.mean(combined_map,axis=2),[i],model_checkpoint)
        imsave('map_{}.png'.format(i), map_style)

        plt.imshow(map_style)
        plt.show()


if __name__ == "__main__":
    main()
