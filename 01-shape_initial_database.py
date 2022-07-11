"""Initial database recognition"""
# In[ ]:
from ast import In

from matplotlib.pyplot import title
import findspark
findspark.init()
import pyspark

# In[ ]:
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")

# %%
df.shape
df.head()

# In[ ]:
df = df.sample(frac=1)

# In[ ]:
df.dtypes

# In[ ]:
df.values

# In[ ]:
type(df.values)

# In[ ]:
df.values.shape

"""Filter data with pandas"""
# In[ ]:
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
df.info()

# In[ ]:
df["ArrDelay"].head()
df[95:100]

# In[ ]:
df[df["ArrDelay"] > 60].head()
df[(df["Origin"]=="ATL") & (df["ArrDelay"] > 60)].head()

# In[ ]:
df[pd.isna(df["ArrDelay"])]

"""transformation"""
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
df.info()

# In[ ]:
df["HoursDelay"] = round(df["ArrDelay"] / 60)
df["HoursDelay"].head()
# In[ ]:
del(df["HoursDelay"])

# In[ ]:
df = df.drop(columns = ["Diverted", "Cancelled", "Year"], axis=1)

# In[ ]:
df = df.drop(range(0,1000000))

# In[ ]:
df.shape

# In[ ]:
df_atl = df[df.Origin == "ATL"]
df_hou = df[df.Origin == "HOU"]
df_all = df_atl.append(df_hou)
df_all.shape
# In[ ]:
"""group by"""
# In[ ]:
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
# In[ ]:
( 
    df.groupby(by="DayOfWeek")[["ArrDelay", "DepDelay"]].max() 
    - df.groupby(by="DayOfWeek")[["ArrDelay", "DepDelay"]].min() 
)

# In[ ]:
df_atl_hou = df[df.Origin.isin(["ATL", "HOU"])]
df_atl_hou.groupby(by = ["DayofWeek", "Origin"])[["ArrDelay", "DepDelay"]].max()

"""drop duplicates"""
# In[ ]:
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv", nrows=1e6)
df_duplicate = df.append(df)
df_duplicate = df_duplicate.sample(frac=1)
df_duplicate.shape
df_clean = df_duplicate.drop_duplicates()
df_clean.shape

df_clean = df_clean.drop_duplicates(subset = ["DayOfMonth"])
df_clean.head()

# In[ ]:
df_clean = df_clean.drop_duplicates(subset = ["DayofMonth"])
df_clean.shape

# In[ ]:
"""drop na values, by the whole row or by specific column"""
df.dropna()
df_clean = df.dropna(thresh = 25)
df_clean = df.dropna(thresh = df.shape[1]-1)
df_clean = df.dropna(subset = ["CancellationCode"])
df_clean.shape


# In[ ]:
"""arrays in numpy"""
import numpy as np
valoraciones = np.array([[1,2,3], [3,2,1], [3,4,1]]) 
valoraciones[0][1]

# In[ ]:
import numpy as np
valoraciones = np.array([
    [
        [1,2,3], 
        [2,2,2]
    ], 
    [
        [2,2,2], 
        [3,2,1]
    ], 
    [
        [1,80,3], 
        [2,2,2]
    ], 
    [
        [2,2,2], 
        [3,2,1]
    ]
])
valoraciones[0,0,0]
# In[ ]:
"""create matrix with default values"""
matrix_zeros = np.zeros((4, 4, 4))
matrix_ones = np.ones((4, 4, 4))
matrix_ones

# In[ ]:
"""np functions"""
np.mean(valoraciones)
np.mean(valoraciones, axis=0)
np.mean(valoraciones, axis=1)
np.mean(valoraciones, axis=2)

# In[ ]:
np.argmax(valoraciones)
# In[ ]:
np.argmin(valoraciones)
# In[ ]:
np.argsort(valoraciones, axis=0)
# In[ ]:
np.reshape([1,2,1,2,2,1,2,1], (2,2,2))

# In[ ]:
np.random.rand(200)

np.random.rand(2,2,2)