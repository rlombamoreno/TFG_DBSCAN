import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.ctypeslib import ndpointer
import ctypes


def Load_image():
    if len(sys.argv) != 2:
        print("serialDBSCAN: Must specify one image filename")
        print("example: python3 serialDBSCAN filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    return np.array(image_orig, dtype = int)

image_bw = Load_image()
print(image_bw)