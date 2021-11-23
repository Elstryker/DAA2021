from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import auxiliary as aux  
import dataSetProcessing as DSProcess
from numpy import ravel
import matplotlib.pyplot as plt

def train_model():
    df = pd.read_csv("training_data.csv",encoding='latin1')
    
    df = DSProcess.treat_dataset(df)
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()
    
    # Instantiate model with 1000 decision trees
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)

    rf = RandomForestClassifier(criterion='entropy',n_estimators=1000)
    # param_grid = {'n_estimators' : [10,100,1000], 'criterion': ['gini','entropy'], }
    # grid = GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=3)

    # grid.fit(x_train,ravel(y_train))

    # print(grid.best_estimator_)

    # print(grid.best_params_)

    # print(grid.best_score_)

    # plot_confusion_matrix(grid,x_test,ravel(y_test))

    # plt.show()


    # Train the model on training data
    # scores = cross_val_score(rfc,x,ravel(y),cv=10)
    # print(scores)
    # print("Result: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(),scores.std()))
    
    rf.fit(x, ravel(y))
    
    X_test = pd.read_csv("test_data.csv",encoding='latin1')
    
    X_test = DSProcess.treat_dataset(X_test)
    
    return rf.predict(X_test)


r = train_model()

aux.to_csv(r)
