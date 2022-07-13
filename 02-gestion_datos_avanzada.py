# In[ ]:
from turtle import title
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv", nrows=400000)
df.to_csv('base_datos_2008.csv')

# In[ ]:
"""np.corrcoef"""
import numpy as np
df = df.dropna(subset = ["ArrDelay", "DepDelay"])
np.corrcoef([df["ArrDelay"], df["DepDelay"]])

# %%
"""np.corr"""
df.drop(inplace=True, columns=["Year", "Cancelled", "Diverted"])
df.corr()

# In[ ]:
df.drop(inplace=True, columns = ["Month"])
corr = round(df.corr(), 3)
corr.style.background_gradient()

# In[ ]:
"""chi-squeared"""
