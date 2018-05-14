import csv
import cv2
import numpy as np

lines = []

#Read Driving_log.csv info
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skip header line
    for line in reader:
        lines.append(line)

#Divide csv data into appropriate lists
imagepaths = []
measurements = []
steering_correction = 0.225 #PARAMETER: correction for left/right camera
steering_correction_multiplier = [0.0,1.0,-1.0] #CENTER,LEFT,RIGHT
for line in lines:
    for img_index in range(3):
        #img_index 0=center camera, 1=left camera, 2=right camera
        source_path = line[img_index]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
        #image = cv2.imread(current_path)
        imagepaths.append(current_path)
        measurement = float(line[3]) + (steering_correction_multiplier[img_index] * steering_correction)
        measurements.append(measurement)
    
#Setup and Pre-process data
X_train = np.array(imagepaths)
y_train = np.array(measurements)

#split to training/validation sets
import sklearn
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20)

#Generator
def generator(X_samples, Y_samples, batch_size=32):
    num_samples = len(X_samples)
    assert num_samples == len(Y_samples)
    
    debug = True
    
    # of Augments
    augments = 1
    true_batch_size = int(batch_size / (1 + augments));
    
    while 1:
        sklearn.utils.shuffle(X_samples, Y_samples) ##TODO Review this
        
        for offset in range(0, num_samples, true_batch_size):
            X_Batch = X_samples[offset:offset+true_batch_size]
            Y_Batch = Y_samples[offset:offset+true_batch_size]
            
            images = []
            angles = []
            for x_batch_sample_name, y_batch_sample in zip(X_Batch, Y_Batch):
                #Read original image
                image = cv2.imread(x_batch_sample_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(y_batch_sample)

                images.append(image)
                angles.append(angle)
                
                #Augmented image (flip on X)
                flip_image = cv2.flip(image,1)
                flip_angle = angle*-1.0
                
                images.append(flip_image)
                angles.append(flip_angle)
                
                if not debug:
                    debug = True
                    cv2.imwrite("test1.jpg", image)
                    cv2.imwrite("test1_flipped.jpg", flip_image)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

#HYPER PARAMS
batch_size = 128
keep_rate = 0.5

train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_validation, y_validation, batch_size=batch_size)

#Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

#Architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_rate))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_rate))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_rate))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Train
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train), validation_data=validation_generator, nb_val_samples = len(X_validation), nb_epoch=7, verbose=1)

model.save('model.h5')

#Plot the training and validation loss for each epoch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./Figure.png')