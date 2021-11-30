from numpy.random.mtrand import random_sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import auxiliary as aux  
import dataSetProcessing as DSProcess
from numpy import ravel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
import sklearn

RANDOM_SEED = 2021
DIMENSION_MODEL = 17

def build_model(optimizer='adam', activation="relu"):
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=DIMENSION_MODEL, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    return model

def train_model():
    
    df = pd.read_csv("treated.csv",encoding='latin1')
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()



    scaler_x = MinMaxScaler(feature_range=(0,1)).fit(x)
    x_scaled = pd.DataFrame(scaler_x.transform(x[x.columns]), columns = x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.20,random_state=RANDOM_SEED)

    y_test = tf.keras.utils.to_categorical(y_test)
    y_train = tf.keras.utils.to_categorical(y_train)

    model = build_model()
    #param_grid = {'n_estimators' : [10,100,1000], 'criterion': ['gini','entropy'], }
    #grid = GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=3)
    model.fit(x_train, y_train,epochs = 10)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)


    return model


clf = train_model()

#df = pd.read_csv("training_data.csv",encoding='latin1')
#df = DSProcess.treat_dataset(df)
#df.to_csv('treated.csv',index=False)

X_test = pd.read_csv("test_data.csv",encoding='latin1')

X_test = DSProcess.treat_dataset(X_test)

scaler_x_test = MinMaxScaler(feature_range=(0,1)).fit(X_test)
x_scaled_test = pd.DataFrame(scaler_x_test.transform(X_test[X_test.columns]), columns = X_test.columns)

r = clf.predict(x_scaled_test)

classes=['None','Low','Medium','High','Very_High']

predictions=[]

for i,pred in enumerate(r):
    prediction = np.argmax(pred)
    predictions.append(classes[prediction])

print(predictions)

aux.to_csv(predictions)
