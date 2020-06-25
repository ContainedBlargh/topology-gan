import os, sys
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from gan import Gan
from math import sqrt
import pickle


def load_images_from_folder(folder):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith("png")]
    print(f"found {len(onlyfiles)} images...")

    training_data = []
    for _file in onlyfiles:
        img = load_img(f"{folder}/{_file}", color_mode='grayscale')
        x = img_to_array(img)[:, :, 0] / 256.0
        training_data.append(x)

    return np.array(training_data)


def main(argv):
    if len(argv) != 2:
        print("Usage of this program:\npython main.py <path to images folder>")
        return
    folder = argv[1]

    training_data = None

    if not os.path.isfile(folder + ".pickle"):
        training_data = load_images_from_folder(folder)
        pickle.dump(training_data, open(folder + ".pickle", "wb"))
    else:
        training_data = pickle.load(open(folder + ".pickle", "rb"))
    print(f"loaded {len(training_data)} images as numpy array.")
    gan = None

    if os.path.isfile("gan.model"):
        gan = pickle.load(open("gan.model", "rb"))
    else:
        gan = Gan()
    gan.train(training_data, 1000)


if __name__ == '__main__':
    main(sys.argv)
