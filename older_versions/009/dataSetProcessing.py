import pandas as pd
from pandas.core.frame import DataFrame
import auxiliary as aux
import holidays    
import numpy as np  


def remove_outliers(df : DataFrame, category):
    # IQR
    Q1 = np.percentile(df[category], 25,
                    interpolation = 'midpoint')

    Q3 = np.percentile(df[category], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1

    # Above Upper bound
    df.loc[(df[category] > Q3+1.5*IQR),[category]] = Q3+1.5*IQR

    # Below Lower bound
    df.loc[(df[category] < Q1-1.5*IQR),[category]] = Q1-1.5*IQR

    return df



def treat_dataset(df : DataFrame):

    #Todos os valores são iguais nessas colunas portanto podemos remove-las
    df = df.drop(["city_name","AVERAGE_PRECIPITATION"],axis=1)
    
    #treating AVERAGE_CLOUDINESS column for names that mean the same
    aux.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nuvens quebrados"], "nuvens quebradas")
    aux.replace_values_column(df, "AVERAGE_CLOUDINESS", ["nublado","tempo nublado"], "céu nublado")
    aux.replace_values_column(df, "AVERAGE_CLOUDINESS",["céu claro"], "céu limpo")
    aux.replace_values_column(df, 'AVERAGE_CLOUDINESS', "NULL", "céu pouco nublado")
    
    #Transforming categorical values into ordinal ones
    df['AVERAGE_CLOUDINESS'] = df['AVERAGE_CLOUDINESS'].map({"céu limpo": 0, 
                                                            "céu pouco nublado" : 1, 
                                                            "algumas nuvens": 2, 
                                                            "nuvens dispersas" : 3, 
                                                            "nuvens quebradas" : 4, 
                                                            "céu nublado" : 5})
    
    #Removing AVERAGE_RAIN since it's always missing
    df = df.drop("AVERAGE_RAIN",axis=1)
    
    #LUMINOSITY column to numerical values
    df['LUMINOSITY'] = df['LUMINOSITY'].map({"DARK": 0, "LOW_LIGHT": 1, "LIGHT": 2})


    cols = df.columns.tolist()

    #Turning date string to datetime
    df['record_date'] = pd.to_datetime(df['record_date'])
     
    #Initiating new columns
    df['weekend'],df['holiday'],df['morning'],df['lunch_time'],df['afternoon'],df['dinner_time'],df['evening'],df['early_morning'] = ([None] * len(df) for i in range(8) )
    
    for i,date_time in enumerate(df['record_date']):
        weekend = int(date_time.weekday()>4)
        year = date_time.year
        date = str(date_time.day) + "-" + str(date_time.month) + "-" + str(year)
        portugal_holidays = holidays.Portugal(years=year)
        holiday = int(date in portugal_holidays)
        
        #date_times for comparison
        morning_start = date_time.replace(hour=7, minute=0)
        lunch_start = date_time.replace(hour=12, minute=0)
        noon_start = date_time.replace(hour=14, minute=0)
        dinner_start = date_time.replace(hour=19, minute=0)
        evening_start = date_time.replace(hour=21, minute=0)
        early_morning_start = date_time.replace(hour=0, minute=0)
        
        morning,lunch_time,afternoon,dinner_time,evening,early_morning=(0 for i in range(6))
        
        if lunch_start > date_time >= morning_start:
            morning=1
        elif noon_start > date_time >= lunch_start:
            lunch_time=1
        elif dinner_start > date_time >= noon_start:
            afternoon=1
        elif evening_start > date_time >= dinner_start:
            dinner_time=1
        elif early_morning_start > date_time >= evening_start:
            evening=1
        elif morning_start > date_time >= early_morning_start:
            early_morning=1
        
        df.loc[i,'weekend'] = weekend
        df.loc[i,'holiday'] = holiday
        df.loc[i,'morning'] = morning
        df.loc[i,'lunch_time'] = lunch_time
        df.loc[i,'afternoon'] = afternoon
        df.loc[i,'dinner_time'] = dinner_time
        df.loc[i,'evening'] = evening
        df.loc[i,'early_morning'] = early_morning
    
    
    df = df.drop("record_date",axis=1)

    if 'AVERAGE_SPEED_DIFF' in cols:
        cols.remove('AVERAGE_SPEED_DIFF')
    cols.remove('record_date')
    cols.remove('AVERAGE_TIME_DIFF')

    for cat in cols:
        df = remove_outliers(df,cat)


    return df