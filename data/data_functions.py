import os
import re
import numpy as np

def filter_path(images_path, annotations_path):
    images_basepath = [re.sub('_fake', '', os.path.basename(i)) for i in images_path]
    annotations_basepath = [os.path.basename(i) for i in annotations_path]
    img_idx = [True if item in annotations_basepath else False for item in images_basepath]
    annotations_idx = [True if item in images_basepath else False for item in annotations_basepath]
    images_path = np.array(images_path)[img_idx]
    annotations_path = np.array(annotations_path)[annotations_idx]
    return images_path, annotations_path