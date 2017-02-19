import csv
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Cropping2D, Input, Lambda, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image


# Batch Generator
def generate_batchs_from_log(data_log, batch_size):
    batch_x = []
    batch_y = []
    while 1:
        X_train_log, y_train_log = shuffle(data_log['images_path'], data_log['angles'])
        for img, angle in zip(X_train_log, y_train_log):
            # add images and angles to batches
            batch_x.append(np.asarray(Image.open(img)))
            batch_y.append(angle)

            if len(batch_x) == batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


# Resize Image
def resize_im(x, size, mthd):
    import tensorflow as tf
    return tf.image.resize_images(x, size, method=mthd)


BATCH_SIZE = 128

images_path = []
angles = []
with open('./recorder_data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.15    # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        # add images path and angles to data set
        directory = "./recorder_data/"
        images_path.extend([directory + row[0], directory + row[1], directory + row[2]])
        angles.extend([steering_center, steering_left, steering_right])

# split data into training set and testing set
x_train, x_val, y_train, y_val = train_test_split(images_path, angles, test_size=0.3, random_state=20)
train_data_log = {'images_path': x_train, 'angles': y_train}
val_data_log = {'images_path': x_val, 'angles': y_val}
# model definition
input_tensor = Input(shape=(160, 320, 3))
# set up cropping2D layer
crop_input = Cropping2D(cropping=((50, 20), (0, 0)))(input_tensor)  # 90, 320, 3
# set up normalization layer
norm_input = Lambda(lambda x: (x / 127.5) - 1)(crop_input)        # 90, 320, 3
# set up resizing layer
resize_input = Lambda(resize_im,
                      output_shape=(224, 224, 3),
                      arguments={'size': (224, 224),
                                 'mthd': 1})(norm_input)

# InceptionV3 excluding Output Layer
base_inception = InceptionV3(include_top=False, input_tensor=resize_input)
for layer in base_inception.layers:
    layer.trainable = False
inception_output = base_inception.output

flatten_inception = Flatten()(inception_output)
flatten_inception = Dropout(0.5)(flatten_inception)
dense_1 = Dense(1024)(flatten_inception)
dense_1 = Dropout(0.5)(dense_1)
angle_predict = Dense(1)(dense_1)

model = Model(input=input_tensor, output=angle_predict)
model.compile(optimizer='adam', loss='mean_squared_error')

history_object = model.fit_generator(generate_batchs_from_log(train_data_log, BATCH_SIZE),
                                     samples_per_epoch=len(x_train),
                                     validation_data=generate_batchs_from_log(val_data_log, BATCH_SIZE),
                                     nb_val_samples=len(x_val),
                                     nb_epoch=10, verbose=1)

model.save('model.h5')
