import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# View first n values of the file
def head(df : DataFrame, num = 5):
    print(df.head(num))

# View info about the categories
def info(df : DataFrame):
    print(df.info())

# Calculate info like count, mean, etc, of the various categories
def describe(df : DataFrame, cat = ''):
    if cat == '':
        print(df.describe())
    else:
        print(df[cat].describe())

# See if there is missing values
def anyMissingValues(df : DataFrame):
    print(df.isna().any())

# Calculate the sum of missing values
def missingValuesSum(df : DataFrame):
    print(df.isna().sum())

# Get unique values of category
def unique(df : DataFrame, cat, size=False):
    if size:
        print(df[cat].unique().size)
    else:
        print(df[cat].unique())

# Grouping categories
def groupBy(df : DataFrame, cat, method = 'count'):
    if method == 'mean':
        print(df.groupby(by=cat).mean())
    elif method == 'count':
        print(df.groupby(by=cat).count())
    elif method == 'sum':
        print(df.groupby(by=cat).sum())
    else:
        pass
        
def replace_values_column(df : DataFrame, col, to_replace, new_value ):
    if to_replace == "NULL":
        df[col].fillna(new_value, inplace = True)
    df[col] = df[col].replace(to_replace,new_value)
    return df

def to_csv(predictions,name="./output.csv"):
    f = open(name, "a")
    f.write("RowId,Speed_Diff\n")
    for index,prediction in enumerate(predictions):
        f.write(str(index+1)+","+prediction+"\n")
    f.close()
        
    