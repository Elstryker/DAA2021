from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import seaborn as sns
from pylab import *
import pandas as pd
import numpy as np
import random
import glob
import cv2
%matplotlib inline

def put_pictures_in_base_folder():
    folders = ['dogs','cats','panda']
    for folder in folders:
        for file in glob.glob("animals/test_data/" + folder + "/*"):
            Path(file).rename(file.replace("test_data", "train_data"))
    
def put_pictures_in_test(percentage_test):
    folders = ['dogs','cats','panda']
    number_files_to_move = int(percentage_test/100 * 1000)
    for folder in folders:
        image_numbers = set()
        for _ in range(number_files_to_move):
            image_number = random.randint(1,1000)
            while image_number in image_numbers:
                image_number = random.randint(1,1000)
            image_numbers.add(image_number)
            image = str(image_number)
            missing = 5 - len(image) # getting number of 0's to append
            image = folder + "_" + "0"*missing + image + ".jpg"
            Path("animals/train_data/" + folder + "/" + image).rename("animals/test_data/" + folder + "/" + image)
    
def prepare_dataset():
    #put all pictures in training
    put_pictures_in_base_folder()
    #choose given percentage of pictures for testing
    put_pictures_in_test(20)
    
    #geradores de imagem para que os valores de RGB fiquem todos entre 0 e 1
    train = ImageDataGenerator(rescale=1/255)
    validation = ImageDataGenerator(rescale=1/255)

    #Creating datasets for training and testing 75 to 25 percent
    train_dataset = train.flow_from_directory("animals/train_data",
                                              target_size= (200,200),
                                              batch_size = 300,
                                              class_mode = 'categorical')

    validation_dataset = train.flow_from_directory("animals/test_data",
                                              target_size= (200,200),
                                              batch_size = 300,
                                              class_mode = 'categorical')
    return train_dataset,validation_dataset

   
#Build convolutional networks model
def build_model(activation="relu"):
    # create model
    model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16, (3,3), activation = activation, input_shape=(200,200,3)),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(32, (3,3), activation = activation),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(64, (3,3), activation = activation),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(128, (3,3), activation = activation),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(81, activation = activation),
                                        tf.keras.layers.Dense(27, activation = activation),
                                        tf.keras.layers.Dense(9, activation = activation),
                                        tf.keras.layers.Dense(3, activation = 'softmax')
        ]
        )
    # Compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(0.001),
                  metrics = ['accuracy' ])
    return model

#Train model 
def train_model(model,train_dataset,validation_dataset):
    model_fit = model.fit(train_dataset,
        validation_data = validation_dataset,
        epochs = 15,
        steps_per_epoch = 5,
        verbose = 1)
    return model

def plot_learning_curve(history, metric):
    plt.figure()
    plt.title('Training ' + metric +' vs Validation ' + metric)
    plt.plot(history.epoch, history.history[metric], label= 'train')
    plt.plot(history.epoch, history.history['val_' + metric], label= 'val')
    plt.ylabel('Training ' + metric)
    plt.xlabel('Epochs')
    plt.legend()
    

train_dataset, validation_dataset = prepare_dataset()
model = build_model()
model = train_model(model,train_dataset,validation_dataset)
plot_learning_curve(model.history, metric= 'loss')
plot_learning_curve(model.history, metric= 'accuracy')

