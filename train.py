import csv
import cv2
import numpy as np
import gc

gc.collect()

from keras.models import Sequential
from keras.layers import Flatten,Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import Lambda, Cropping2D

BATCH_SIZE = 32
EPOCHS = 5

#data_dir = "sim_data/"
data_dir = "data/"
log_file = "driving_log.csv"
image_dir = data_dir + "IMG/"

#
# model = Sequential()
# model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# model.add(Conv2D(nb_filter=6,nb_col=5,nb_row=5,activation="relu"))
# #model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(nb_filter=6,nb_col=5,nb_row=5,activation="relu"))
# #model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# #model.add(Dropout(0.2))
# model.add(Dense(84))
# #model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mse',optimizer='adam')

lines = []
first_line=True
with open(data_dir + log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (first_line):
            first_line = False
            continue
        lines.append(line)

center_images = []
steering_measurements = []
for line in lines:
    # parse the line
    center_img_path = line[0]
    left_img_path   = line[1]
    right_img_path  = line[2]
    steering = float(line[3])
    throttle = float(line[4])
    brake = float(line[5])
    speed = float(line[6])
    # Set relative directories to use
    center_img_filename = image_dir + center_img_path.split('/')[-1]
    left_img_filename   = image_dir + left_img_path.split('/')[-1]
    right_img_filename  = image_dir + right_img_path.split('/')[-1]
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
    # TODO : append left and right images


X_val = np.array(center_images)
y_val = np.array(steering_measurements)

print((np.shape(center_images), np.shape(steering_measurements)))
print((X_val.shape, y_val.shape))


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(6,5,5,activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.2))
model.add(Dense(84))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_val,y_val,nb_epoch=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.3,shuffle=True)


model.save('model.h5')