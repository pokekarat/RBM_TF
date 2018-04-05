from scipy import misc
import tensorflow as tf
import numpy as np


def sample_prob(probs, rand):
    """ Takes a tensor of probabilities (as from a sigmoidal activation)
    and samples from all the distributions
    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values
    :return : binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))



