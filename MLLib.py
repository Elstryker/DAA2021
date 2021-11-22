from pandas.core.frame import DataFrame

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
    results = df.isna().any()
    print(results)
    return df.isnull().values.any() #retorna true se algum valor está em falta no dataset recebido

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