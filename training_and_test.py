import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Download MNIST data """
    # Remake path
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    # Load files
    with open(labels_path, 'rb') as lbpath:
        # Change binary to string
        magic, n = struct.unpack('>II', lbpath.read(8))
        # Load label and create array
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # Change binary to array
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images,labels
