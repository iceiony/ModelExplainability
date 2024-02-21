import tests
import pytest

from importlib import reload;
import pandas as pd
import numpy as np

import mnist as mn


def test_can_load_mnist_train_data_and_labels:
    images, labels = mn.load_mnist_data('./data')

    assert images.shape == (60000, 28, 28)
    assert labels.shape == (60000,)

    #import matplotlib.pyplot as plt

    #plt.ion()
    #plt.imshow(images[1])
