import pandas as pd
import MLLib

df = pd.read_csv("training_data.csv")

MLLib.unique(df,"LUMINOSITY",True)

MLLib.groupBy(df,['LUMINOSITY'],method='sum')