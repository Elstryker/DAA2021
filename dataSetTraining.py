from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import auxiliary as aux  
import dataSetProcessing as DSProcess

def train_model():
    df = pd.read_csv("training_data.csv",encoding='latin1')
    
    df = DSProcess.treat_dataset(df)
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()
    
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000)

    # Train the model on training data
    rf.fit(x, y);
    
    X_test = pd.read_csv("test_data.csv",encoding='latin1')
    
    X_test = DSProcess.treat_dataset(X_test)
    
    return rf.predict(X_test)


r = train_model()

aux.to_csv(r)