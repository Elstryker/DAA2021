import pandas as pd
import MLLib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def treat_dataset(df):
    #Todos os valores são iguais nessas colunas portanto podemos remove-las
    df = df.drop(["city_name","AVERAGE_PRECIPITATION"],axis=1)
    
    #a coluna AVERAGE_CLOUDINESS tinha valores repetidos mas com nomes diferentes portanto foi tratada
    
    df = MLLib.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nuvens quebrados"], "nuvens quebradas")
    
    df = MLLib.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nublado","tempo nublado"], "céu nublado")
    
    df = MLLib.replace_values_column(df,"AVERAGE_CLOUDINESS",["céu claro"], "céu limpo")
    
    df = MLLib.replace_values_column(df, 'AVERAGE_CLOUDINESS', "NULL", "céu pouco nublado")
    
    #transformar os valores categoricos em valores numéricos
    
    df = df.replace("céu limpo",0)
    
    df = df.replace("céu pouco nublado",1)
    
    df = df.replace("algumas nuvens",2)
    
    df = df.replace("nuvens dispersas",3)
    
    df = df.replace("nuvens quebradas",4)
    
    df = df.replace("céu nublado",5)
    
    #a coluna AVERAGE_RAIN tinha valores repetidos mas com nomes diferentes portanto foi tratada
    
    MLLib.replace_values_column(df, "AVERAGE_RAIN", ["trovoada com chuva","trovoada com chuva leve"], "trovoada")
    
    MLLib.replace_values_column(df, "AVERAGE_RAIN", ["aguaceiros","chuva"],"chuva moderada")
    
    MLLib.replace_values_column(df, "AVERAGE_RAIN", ["chuva leve","chuvisco e chuva fraca","chuvisco fraco","aguaceiros fracos",], "chuva fraca")
    
    MLLib.replace_values_column(df, "AVERAGE_RAIN", ["chuva de intensidade pesado","chuva de intensidade pesada"], "chuva forte")
    
    MLLib.replace_values_column(df, 'AVERAGE_RAIN', "NULL", "sem chuva") #existem tantos nulos que talvez seja melhor remover a coluna
    
    #transformar os valores categoricos em valores numéricos
    
    df = df.replace("sem chuva",-1)

    df = df.replace("chuva fraca",0)
    
    df = df.replace("chuva moderada",1)
    
    df = df.replace("chuva forte",2)
    
    df = df.replace("trovoada",4)
    
    #LUMINOSITY column to numerical values
    
    df = df.replace("DARK",-1)
    df = df.replace("LOW_LIGHT",0)
    df = df.replace("LIGHT",1)


    ##
    ## DROPPING DATE JUST FOR TEST
    ##

    df = df.drop("record_date",axis=1)
    
    
    
    #O seguinte código permite uma melhor avaliação dos valores de cada coluna pois agrupa com base em cada coluna
    grouped_by_data = []
    columns = df.columns

    for column in columns:
        grouped_by_data.append(df.groupby(column).count())
    
    return df



def train_model():
    df = pd.read_csv("training_data.csv",encoding='latin1')
    
    df = treat_dataset(df)
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()
    
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.05,random_state=2021)
    
    X_test = pd.read_csv("test_data.csv",encoding='latin1')
    
    X_test = treat_dataset(X_test)
    
    clf = DecisionTreeClassifier(random_state=2021)
    
    clf.fit(X_train,Y_train)
    
    return clf.predict(X_test)



df = pd.read_csv("training_data.csv",encoding='latin1')
 
df = treat_dataset(df)


r = train_model()

MLLib.to_csv(r)



