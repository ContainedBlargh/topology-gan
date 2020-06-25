import os, sys
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from gan import Gan


def main():
    folder = "B:\\Source\\Kotlin\\MarsTopologyCollector\\terrain"
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print(f"found {len(onlyfiles)} images...")

    training_data = []
    for _file in onlyfiles:
        img = load_img(f"{folder}/{_file}")
        img.convert('LA')
        x = img_to_array(img)

        training_data.append(x)

    training_data = np.array(training_data)

    gan = Gan()
    gan.train(training_data, 100)


if __name__ == '__main__':
    main()
