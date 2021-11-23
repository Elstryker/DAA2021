from pandas.core.frame import DataFrame

# Replacing values on a certain column
def replace_values_column(df : DataFrame, col, to_replace, new_value ):
    if to_replace == "NULL":
        df[col].fillna(new_value, inplace = True)
    df[col] = df[col].replace(to_replace,new_value)
    return df

# Write predictions to csv
def to_csv(predictions,name="./output.csv"):
    f = open(name, "w")
    f.write("RowId,Speed_Diff\n")
    for index,prediction in enumerate(predictions):
        f.write(str(index+1)+","+prediction+"\n")
    f.close()
        
    