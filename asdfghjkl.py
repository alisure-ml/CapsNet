from scipy.io import loadmat, savemat
from glob import glob
from PIL import Image
import numpy as np
import os


def to_mat(image_path, label_file, result_file):

    x = []
    image_files = glob(os.path.join(image_path, "*.bmp"))
    for image_file in image_files:
        image_data = np.copy(np.asarray(Image.open(image_file)))
        x.append(image_data)
        pass

    # [size, size, 3, image_number]
    x = np.transpose(x, axes=[3, 0, 1, 2])
    y = loadmat(label_file)["y"]

    savemat(result_file, mdict={"x": x, "y": y})
    pass

if __name__ == '__main__':
    to_mat("img/train", "img/label_train.mat", "train_256x256.mat")
    to_mat("img/test", "img/label_test.mat", "test_256x256.mat")
