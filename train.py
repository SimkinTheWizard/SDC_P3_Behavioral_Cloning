import csv
import cv2
import numpy as np
import gc

gc.collect()

from keras.models import Sequential
from keras.layers import Flatten,Dense, Conv2D, MaxPooling2D, Dropout, Activation
from keras.layers import Lambda, Cropping2D

import sklearn
from sklearn.model_selection import train_test_split


BATCH_SIZE = 32
EPOCHS = 5

# pre provided data
data_dir = "data/"
# generated from simulation
sim_data_dir = "sim_data/"
# second simulation : driving from sides
sim_data_sides_dir = "sim_data_s/"
# default_file_names
log_file = "driving_log.csv"
image_dir = data_dir + "IMG/"


def get_file_names(data_dir):
    lines = []
    first_line=True
    with open(data_dir + log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (first_line):
                first_line = False
                continue
            lines.append(line)
    return lines

def read_description_files():
    lines = []
    for line in get_file_names(data_dir):
        lines.append(line)

    for line in get_file_names(sim_data_dir):
        lines.append(line)

    for line in get_file_names(sim_data_sides_dir):
        lines.append(line)

    return lines

def get_relative_file_name(absolute_file_name):
    components = absolute_file_name.split('/')
    if (len(components) > 2):
        return components[-3] + "/" + components[-2] + "/" + components[-1]
    else:
        return data_dir + components[-2] + "/" + components[-1]

def generate_data(lines,batch_size=BATCH_SIZE):
    num_samples = len(lines)
    while (True):
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            center_images = []
            steering_measurements = []

            for line in lines[offset:offset+batch_size]:
                # parse the line
                steering = float(line[3])
                throttle = float(line[4])
                brake = float(line[5])
                speed = float(line[6])
                # Set relative directories to use
                center_img_filename = get_relative_file_name(line[0])
                left_img_filename   = get_relative_file_name(line[1])
                right_img_filename  = get_relative_file_name(line[2])
                # append the values
                center_images.append(cv2.imread(center_img_filename))
                steering_measurements.append(steering)
                # TODO : append left and right images
                #print(center_img_filename)
                #image augmentation
                center_image_flipped = np.fliplr(cv2.imread(center_img_filename))
                measurement_flipped = -steering
                center_images.append(center_image_flipped)
                steering_measurements.append( measurement_flipped)
                # TODO : append steering left and right images
                X_val = np.array(center_images)
                y_val = np.array(steering_measurements)
                yield sklearn.utils.shuffle(X_val, y_val)



def initialize_model():
    model = Sequential()
    # Preprocessing
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # Conv 1 6 *  5x5
    model.add(Conv2D(6,5,5,activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    # Conv 2 12 *  5x5
    model.add(Conv2D(12,5,5,activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    # Conv 3 18 *  3x3
    model.add(Conv2D(6,3,3,activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    # Conv 4 24 *  3x3
    model.add(Conv2D(6,3,3,activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    # Flatten
    model.add(Flatten())
    # Dense 1
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    # Dense 2
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    # Dense 3
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    # Output
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model


def main():

    lines = read_description_files()
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    #train_samples, validation_samples = train_test_split(sklearn.utils.shuffle(lines), test_size=0.2)

    train_generator = generate_data(train_samples)
    validation_generator = generate_data(validation_samples)

    model = initialize_model()

    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

    model.save('model.h5')

if __name__ == '__main__':
    main()