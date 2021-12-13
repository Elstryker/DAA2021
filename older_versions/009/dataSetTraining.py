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
from keras import callbacks
import sklearn

RANDOM_SEED = 2021
DIMENSION_MODEL = 17

def build_model(learning_rate=0.01, activation="tanh",momentum=0.4,init_mode='glorot_uniform', neurons=20, layers=2):
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=DIMENSION_MODEL,kernel_initializer=init_mode, activation=activation))
    for lay in range(layers):
        model.add(Dense(neurons, input_dim=DIMENSION_MODEL,kernel_initializer=init_mode, activation=activation))
    model.add(Dense(5, activation='softmax',kernel_initializer=init_mode))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = tf.optimizers.Adam(learning_rate), metrics=['accuracy'])
    return model

def train_model():
    
    df = pd.read_csv("treated.csv",encoding='latin1')
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()

    y['AVERAGE_SPEED_DIFF'] = y['AVERAGE_SPEED_DIFF'].map({"None": 0, 
                                                            "Low" : 1, 
                                                            "Medium": 2, 
                                                            "High" : 3, 
                                                            "Very_High" : 4})

    scaler_x = MinMaxScaler(feature_range=(0,1)).fit(x)
    x_scaled = pd.DataFrame(scaler_x.transform(x[x.columns]), columns = x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.15,random_state=RANDOM_SEED)
    
    #y_test = tf.keras.utils.to_categorical(y_test)
    y_train = tf.keras.utils.to_categorical(y_train)
    
    y = tf.keras.utils.to_categorical(y)
    
    
    #model = KerasClassifier(build_fn=build_model,batch_size=40,epochs=100,learning_rate=0.001,momentum=0.4, verbose=2)
    model = build_model()
    #model.fit(x_train,y_train,epochs=100,batch_size=40)
    model.fit(x_scaled,y,epochs=100,batch_size=40)
    
    # define the grid search parameters
    #layers = [1,2,3]
    #neurons = [1, 5, 10, 15, 20, 25, 30, 40]
    #param_grid = dict(layers=layers,neurons=neurons)
    #grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,verbose=3,n_jobs=-1)
    #grid_result = grid.fit(x_scaled,y)
    
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    
    
    #preds = model.predict(x_test)

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    # print('\nTest accuracy:', test_acc)
    
    # predictions=[]

    # for i,pred in enumerate(preds):
    #     prediction = np.argmax(pred)
    #     predictions.append(prediction)
    
    # print("Accuracy Score: ", accuracy_score(y_test['AVERAGE_SPEED_DIFF'].tolist(),predictions))

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
