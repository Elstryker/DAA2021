import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression

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

# Calculate info of the selected category
# print(df["WebActivity"].describe())

# See if there is missing values
# print(df.isna().any())

# Calculate the sum of missing values
# print(df.isna().sum())

# Get unique values of category
# print(df.WebActivity.unique())
# print(df.WebActivity.unique().size)

# Grouping categories
# print(df.groupby(by=['Gender']).mean())
# print(df.groupby(by=['SentimentRating','Gender']).count())