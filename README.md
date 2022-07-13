# learning_linkedin_python_data_science_big_data_esencial

# ds
[repo](https://github.com/programanime/learning_linkedin_python_data_science_big_data_esencial)   
[tutorial](https://www.linkedin.com/learning/python-para-data-science-y-big-data-esencial)  

# big data needs
- depends of the epoch
![](img/22206.png)  
![](img/24356.png)  


# questions before start
1. do we need to use all the available data?
2. do we need to rush on it?
3. is data structured?
4. the current data is static or dynamic?

# in case we really need big data
1. ram
2. disk
3. cpu

# spark
```shell
PYSPARK_DRIVER_PYSON_OPTS=notebook
PYSPARK_DRIVER_PYTHON=jupyter
SPARK_HOME=A:\spark-3.3.0-bin-hadoop3
HADOOP_HOME=A:\spark-3.3.0-bin-hadoop3
```

## configure spark
```python
import findspark
findspark.init()
import pyspark
```


```python
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
```

## data sample
```python
df = df.sample(frac=1)
```

## columns
```python
df.columns
```

## types
```python
df.dtypes
```

## values
 - return the values as a ndarray
 - you just get the matrix of values

## pandas access to rows

```python
df[95:100]
```

## pandas conditions
```python
df[df["ArrDelay"] > 60].head()
```

## not available cases
```python
df[pd.isna(df["ArrDelay"])]
```

# transformation

```markdown
# ds
[repo](https://github.com/programanime/learning_linkedin_python_data_science_big_data_esencial)   
[tutorial](https://www.linkedin.com/learning/python-para-data-science-y-big-data-esencial)  

# big data needs
- depends of the epoch
![](img/22206.png)  
![](img/24356.png)  


# questions before start
1. do we need to use all the available data?
2. do we need to rush on it?
3. is data structured?
4. the current data is static or dynamic?

# in case we really need big data
1. ram
2. disk
3. cpu

# spark
```shell
PYSPARK_DRIVER_PYSON_OPTS=notebook
PYSPARK_DRIVER_PYTHON=jupyter
SPARK_HOME=A:\spark-3.3.0-bin-hadoop3
HADOOP_HOME=A:\spark-3.3.0-bin-hadoop3
```

## configure spark
```python
import findspark
findspark.init()
import pyspark
```


```python
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
```

## data sample
```python
df = df.sample(frac=1)
```

## columns
```python
df.columns
```

## types
```python
df.dtypes
```

## values
 - return the values as a ndarray
 - you just get the matrix of values

## pandas access to rows

```python
df[95:100]
```

## pandas conditions
```python
df[df["ArrDelay"] > 60].head()
```

## not available cases
```python
df[pd.isna(df["ArrDelay"])]
```

# transformation
## adding new column
```python
df["HoursDelay"] = round(df["ArrDelay"] / 60)
df["HoursDelay"].head()
```

## del function
built-in function, let you remove any value from dataframe or dict

## drop just one column
```python
del(df["HoursDelay"])
```

##  drop a bunch of columns
```python
df = df.drop(columns = ["Diverted", "Cancelled", "Year"], axis=1)
```

## drop rows
```python
df = df.drop(range(0,1000000))
```

## join rows
```python
df_atl = df[df.Origin == "ATL"]
df_hou = df[df.Origin == "HOU"]
df_atl.append(df_hou)
```

# group by 
for each "DayOfWeek" get the mean of ["ArrDelay", "DepDelay"]
```python
df.groupby(by="DayOfWeek")["ArrDelay"].mean()
```

```python
df.groupby(by="DayOfWeek")[["ArrDelay", "DepDelay"]].mean()
```

- you could use several aggregation levels
```python
df_atl_hou.groupby(by = ["DayofWeek", "Origin"])[["ArrDelay", "DepDelay"]].max()
```

### drop duplicates
```python
# check if the whole row is duplicate
df_clean = df_clean.drop_duplicates()

# check if there are duplicates just in that column
df_clean = df_clean.drop_duplicates(subset = ["DayofMonth"])
```

# drop na
drop any record with any empty column, drop ,almost the entire data in the majority of cases

- drop if there are any null value in any column
```python
df_clean = df.dropna()
```

- drop if there are less than 25 columns with value
```python
df_clean = df.dropna(thresh = 25)
```

- drop if there are null values in the specific column
```python
df_clean = df.dropna(subset = ["CancellationCode"])
```

# numpy basics

```python
import numpy as np
valoraciones = np.array([[1,2,3], [3,2,1], [3,4,1]]) 
valoraciones[0][1]
```


## np array with three dimensions
 - there could be N dimensions in the array 
```python
import numpy as np
valoraciones = np.array([[[1,2,3], [2,2,2]], [[2,2,2], [3,2,1]], [[1,2,3], [2,2,2]], [[2,2,2], [3,2,1]]])
valoraciones[0,0,0]
```

## create arrays with default values
 - create matrix with zero as default value
```python
matrix_zeros = np.zeros((4, 4, 4))
```

 - create matrix with one as default value
```python
matrix_ones = np.zeros((4, 4, 4))
```

## np axis
 ![](img/44023.png)  
 - axis 0 -> rows
 - axis 1 -> columns
 - axis 2 -> depth

while you go through the axis, it takes more aggregation level


# np functions
- np.mean
```python
np.mean(valoraciones)
```
- np.meadian
- np.sum
- np.std
- np.var
- np.max
- np.min
- np.argmin
- np.argsort
- np.sort
- np.sort_index
- np.sort_values

## np.argmax
return the index for the value
it took the index as a one-dimension array

```python
np.argmax(valoraciones)
np.sum(valoraciones)
np.std(valoraciones)
np.var(valoraciones)
np.max(valoraciones)
np.min(valoraciones)
```

## np argsort
sort and return the index
```python
np.argsort(valoraciones, axis=0)
```

## np reshape
change the dimension keeping the data
```python
np.reshape([1,2,1,2,2,1,2,1], (2,2,2))
```

## np.random
generate random arrays with any dimension

generate one array
```python
np.random.rand(200)
```

generate one matrix
```python
np.random.rand(2,2,2)
```

# correlation
covariance / standard deviation

$ p_{x,y} = \frac{σ_{xy}}{σ_x σ_y}$

$ = \frac{E[(X-u_x)(Y-u_y)]}{σ_x σ_y} $

- could be [0,1]
- near to zero means no relationship
- near to one means good relationship
- greater than 0.6 is a good relationship

$ p_{x,y} $ 

## np.corrcoef
 - this function is useful when you need the correlation between few variables
 - you couldn't have any missing value
    - you need to remove it
    - you need to impute it
 - give you a matrix with all correlations between the given variables
    - symmetric matrix
    - one diagonal
```python
import numpy as np
df = df.dropna(subset = ["ArrDelay", "DepDelay"])
np.corrcoef(df["ArrDelay"], df["DepDelay"])
```
 - if you need to get the correlation between three or more variables, then you need to pass an array as argument

```python
import numpy as np
df = df.dropna(subset = ["ArrDelay", "DepDelay", "DepTime"])
np.corrcoef([df["ArrDelay"], df["DepDelay"], df["DepTime"]])
```

## np.corr
 - calculate the whole correlation matrix for the dataset
 - before to this, you need to make sure, there are not any object type, you should keep just numerical columns

```python
df.drop(inplace=True, columns=["Year", "Cancelled", "Diverted"])
df.corr()
```


```python
df.drop(inplace=True, columns = ["Month"])
corr = round(df.corr(), 3)
corr.style.background_gradient()
```

# chi-squared
- use contingency tables get the relationship
   - let you find relationship between categorical variables
   - compare the suppose  non relation individual events with the real one

$ X^2 = \sum{\frac{(observado_i - esperado_i)^2}{esperado_i}} $

- you can't realize about relationship for specific categories
   - but, you can realize about general relationship

chi-squared test
- crosstab
   - margins = include all counters
   - normalize = normalize the data
```python
import pandas as pd
df  =  pd.read_csv("base_datos_2008.csv")
np.random.seed(0)
df = df[df["Origin"].isin(["HOU", "ATL", "IND"])]
df = df.sample(frac=1)
df = df[0:10000]

df["BigDelay"] = df["ArrDelay"] > 30

observados  = pd.crosstab(index = df["BigDelay"], columns = df["Origin"], margins = True)
observados.head()

from scipy.stats import chi2_contingency
test = chi2_contingency(observados)
test

esperados  = pd.DataFrame(test[3])
esperados

esperados_rel = round(esperados.apply(lambda r: r/df.shape[0] * 100, axis=1), 2)
observados_rel = round(observados.apply(lambda r: r/df.shape[0] * 100, axis=1), 2)

esperados_rel
observados_rel

f"p-value {test[1]}"
```
## conclusion
 - if the p-value < 0.05 the, the two categorical variables are related
 - for this case there is not relation

# analyzing extreme data
- detect too low values and too high values
- need to get the quartiles, Q1, Q3
   - then calculate the interquartile_range
   - then needs to calculate the threshold, low and hight
   - the threshold is used to get the extreme values
      - then calculate the extreme values
- np.mean(x>threshold)
   - useful for the percentage of extreme values
```python
import pandas as pd
import numpy as np
df  =  pd.read_csv("base_datos_2008.csv")
series_delay_cleaned = df["ArrDelay"].dropna()

Q1 = np.percentile(series_delay_cleaned, 25)
Q3 = np.percentile(series_delay_cleaned, 75)
interquartile_range = Q3 - Q1

umbral_superior = Q3 + 1.5*interquartile_range
umbral_inferior = Q1 - 1.5*interquartile_range

np.mean(series_delay_cleaned > umbral_superior)
np.mean(series_delay_cleaned < umbral_inferior)

f"umbral_superior {umbral_superior}"
f"umbral_inferior {umbral_inferior}"

right_extreme = series_delay_cleaned[series_delay_cleaned > umbral_superior]
(series_delay_cleaned.shape, right_extreme.shape)

left_extreme = series_delay_cleaned[series_delay_cleaned < umbral_inferior]
(series_delay_cleaned.shape, left_extreme.shape)

right_extreme.sort_values(ascending=False).head()
left_extreme.sort_values().head()

np.mean(series_delay_cleaned > umbral_superior)
np.mean(series_delay_cleaned < umbral_inferior)
np.mean((series_delay_cleaned > umbral_superior) | (series_delay_cleaned < umbral_inferior))
```

## conclusions
 - there is almost a 1% of outliers
 - the highest value is 1081 minutes, almost 18 hours
 - the lowest value is -91 minutes, the plane arrives 1.5 hours before

# outliers with sklearn
- EllipticEnvelope from sklearn is useful for get rows with outliers values
```python
import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
df  =  pd.read_csv("base_datos_2008.csv")
series_delay_cleaned = df["ArrDelay"].dropna()
outliers = EllipticEnvelope(contamination=0.01)
features = ["DepDelay" ,"TaxiIn" ,"TaxiOut", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay" , "LateAircraftDelay"]
df_clean =df[features].dropna()
outliers.fit(df_clean)
outliers_values = outliers.predict(x)
outliers_values

index_outliers = np.where(outliers_values == -1)[0]
index_outliers.shape[0] / df_clean.shape[0]
```
### conclusions
  - almost one percent of data are outliers

# transform dataframe to relational databases
- this is useful for ETL process, to save the data in different tables
- basically you need to split and drop duplicates for each table 

```python
import pandas as pd
data  = [
   (1, "daniel", 25, 1, "colgate"), 
   (2, "leidy", 23, 1, "colgate"), 
   (3, "nicol", 10, 1, "colgate")
]
labels = ["id", "name", "age", "id_product", "product_name"]
df = pd.DataFrame.from_records(data, columns=labels)

compradores = df.drop_duplicates(subset = "id", keep="first")[["id", "name", "age"]]
products = df.drop_duplicates(subset = "id_product", keep="first")[["id_product", "product_name"]]

compradores
products
```
### conclusions
  - drop_duplicates function is the key to split in different tables
  - if there is not unique identifiers, this could no be possible

```python
import pandas as pd
consumidores = [("a", "movil"), ("b", "portatil"), ("c", "pc"), ("d", "screen"), ("e", "battery"), ("f", "mouse"), ("g", "keyboard"), ("h", "lamp"), ("i", "hat"), ("j", "glass"), ("k", "gloves"), ("l", "banana")]
productores = [("microsoft", "movil"), ("amazon", "portatil"), ("google", "pc"), ("aws", "screen"), ("micro", "battery"), ("la_80", "mouse"), ("tiendas", "keyboard"), ("airline", "lamp")]
consumidores_labels = ["consumidor", "producto"]
productores_labels = ["productor", "producto"]

df_consumidores = pd.DataFrame.from_records(consumidores, columns = consumidores_labels)
df_productores = pd.DataFrame.from_records(productores, columns = productores_labels)

pd.merge(df_consumidores, df_productores, on="producto", how="outer")
pd.merge(df_consumidores, df_productores, on="producto", how="inner")
pd.merge(df_consumidores, df_productores, on="producto", how="left")
pd.merge(df_consumidores, df_productores, on="producto", how="right")

pd.merge(df_consumidores, df_productores, on="producto", how="outer").shape
pd.merge(df_consumidores, df_productores, on="producto", how="inner").shape
pd.merge(df_consumidores, df_productores, on="producto", how="left").shape
pd.merge(df_consumidores, df_productores, on="producto", how="right").shape
```
  ![](img/26058.jpg)  

### conclusions
  - for inner join there could not be nan values
  - for right and left joins, there could be null values
  - for outer there are more values, and the size is bigger

# parallel execution
⚠ you need to run this in a separate file, to make sure it works
- split the execution between a bunch of cores
- if the data volume is low or the core numbers is low
   - the performance could be less than with a single core
```python
import pandas as pd
import numpy as np
df  =  pd.read_csv("base_datos_2008.csv")
columns = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
df_sub = df[columns]
df_sub.head()
df_clean = df_sub.dropna()

def retraso_maximo(fila):
   if not np.isnan(fila).any():
      return columns[fila.index(max(fila))]
   else:
      return "None"

results = []
for fila in df_clean.values.tolist():
   results.append(retraso_maximo(fila))

results

if __name__ == '__main__':
   from joblib import Parallel, delayed
   result = Parallel(n_jobs = 2, backend = "multiprocessing")(map(delayed(retraso_maximo), df_sub.values.tolist()))
   print(result)
```
### conclusions
  - you need to enclose the function in the main condition, to avoid spawned process to run several times
