import pandas as pd
import MLLib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import datetime                                 
import holidays      

def treat_dataset(df):
    #Todos os valores são iguais nessas colunas portanto podemos remove-las
    df = df.drop(["city_name","AVERAGE_PRECIPITATION"],axis=1)
    
    #a coluna AVERAGE_CLOUDINESS tinha valores repetidos mas com nomes diferentes portanto foi tratada
    
    MLLib.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nuvens quebrados"], "nuvens quebradas")
    
    MLLib.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nublado","tempo nublado"], "céu nublado")
    
    MLLib.replace_values_column(df,"AVERAGE_CLOUDINESS",["céu claro"], "céu limpo")
    
    MLLib.replace_values_column(df, 'AVERAGE_CLOUDINESS', "NULL", "céu pouco nublado")
    
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
    
    df['record_date'] = pd.to_datetime(df['record_date'])
    
    if MLLib.anyMissingValues(df['record_date'])==False:  
        weekend_list = []
        holiday_list = []
        morning_list = []
        lunch_time_list = []
        afternoon_list = []
        dinner_time_list = []
        evening_list = []
        early_morning_list = []
        
        
        for date_time in df['record_date']:
            weekend = int(date_time.weekday()>4)
            year = date_time.year
            date = str(date_time.day) + "-" + str(date_time.month) + "-" + str(year)
            holidayss = holidays.Portugal(years=year)
            holiday = int(date in holidayss)
            
            #date_times for comparison
            morning_start = date_time.replace(hour=7, minute=0)
            lunch_start = date_time.replace(hour=12, minute=0)
            noon_start = date_time.replace(hour=14, minute=0)
            dinner_start = date_time.replace(hour=19, minute=0)
            evening_start = date_time.replace(hour=21, minute=0)
            early_morning_start = date_time.replace(hour=0, minute=0)
            
            morning=0
            lunch_time=0
            afternoon=0
            dinner_time=0
            evening=0
            early_morning=0
            
            if date_time>=morning_start and date_time<lunch_start:
                morning=1
            elif date_time>=lunch_start and date_time<noon_start:
                lunch_time=1
            elif date_time>=noon_start and date_time<dinner_start:
                afternoon=1
            elif date_time>=dinner_start and date_time<evening_start:
                dinner_time=1
            elif date_time>=evening_start and date_time<early_morning_start:
                evening=1
            elif date_time>=early_morning_start and date_time<morning_start:
                early_morning=1
            
            weekend_list.append(weekend)
            holiday_list.append(holiday)
            morning_list.append(morning)
            lunch_time_list.append(lunch_time)
            afternoon_list.append(afternoon)
            dinner_time_list.append(dinner_time)
            evening_list.append(evening)
            early_morning_list.append(early_morning)
        
        df['weekend'] = weekend_list 
        df['holiday'] = holiday_list
        df['morning'] = morning_list 
        df['lunch_time'] = lunch_time_list
        df['afternoon'] = afternoon_list
        df['dinner_time'] = dinner_time_list
        df['evening'] = evening_list
        df['early_morning'] = early_morning_list
    df = df.drop("record_date",axis=1)
    df = df.drop("AVERAGE_RAIN",axis=1)
    
    return df



def train_model():
    df = pd.read_csv("training_data.csv",encoding='latin1')
    
    df = treat_dataset(df)
    
    x = df.drop('AVERAGE_SPEED_DIFF',axis=1)
    y = df['AVERAGE_SPEED_DIFF'].to_frame()
    
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000)
    # Train the model on training data
    rf.fit(x, y);
    
    X_test = pd.read_csv("test_data.csv",encoding='latin1')
    
    X_test = treat_dataset(X_test)
    
    return rf.predict(X_test)

r = train_model()

MLLib.to_csv(r)




