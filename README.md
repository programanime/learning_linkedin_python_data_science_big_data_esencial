# ds
[repo](https://github.com/programanime/learning_linkedin_python_data_science_big_data_esencial)   
[tutorial](https://www.linkedin.com/learning/python-para-data-science-y-big-data-esencial)  
[matplotlib](https://matplotlib.org/)

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


# matplotlib

## pie chart
```python
import pandas as pd
import numpy as np
df  =  pd.read_csv("base_datos_2008.csv")
data = np.unique(df.Cancelled, return_counts=True)
data[1].head()
plt.pie(x=data[1], labels=data[0], autopct='%1.1f%%', colors = ["red", "green"], shadow=True, startangle = 90, radius = 1.1)
plt.show()
```
### conclusions
  - pie chart just need x data and labels
  - useful for categorical or numerical values

## plt burble chart
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)
df  =  pd.read_csv("base_datos_2008.csv")
df  = df.sample(frac=1).head(100)
plt.scatter(
   x = df.DayofMonth, 
   y = df.ArrDelay, 
   s = df.Distance, 
   alpha = .3,
   c = df.DayOfWeek.isin([6,7])
)
plt.xlabel("dia del mes")
plt.ylabel("retraso al llegar")
plt.xticks([0,15,30])
plt.yticks([0,10,30])
plt.text(x=1, y=200, s = "mi fly")
plt.show()
```
### conclusions
useful for three dimensions
x, y and size of bubble is the depth

# pie chart
```python
import pandas as pd
import numpy as np
plt.pie(
   x=[2,1,2,2,50,100], 
   labels=["auto", "a", "a", "a", "a", "a"], 
   autopct='%1.1f%%', 
   colors = sns.color_palette("hls", 7), 
   shadow=True, 
   startangle = 90, 
   explode = (0,0,0.2,0,0,0.1),
   labeldistance = 1,
   radius = 1.1
)
plt.legend(labels = ["auto", "a"])
plt.show()
```
### conclusions
  - i does not matter if the label is repeat
  - it takes the summary

# barplot
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df  =  pd.read_csv("base_datos_2008.csv")
ax = sns.barplot(x="DayOfWeek", y="ArrDelay", data=df)
ax.set(xlabel="dia de la semana", ylabel="retraso al llegar")
plt.show()
```

### conclusions
you could a dataframe, indicate the X and Y column name,
then a data object besides with them

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
df  =  pd.read_csv("base_datos_2008.csv")
times = []

for i in np.arange(len(df)):
   times.append(datetime.datetime(year=2008, month=df.Month[i], day=df.DayofMonth[i]))

df["Time"] = times

data = df.groupby("Time", as_index=False).agg({"ArrDelay": "mean", "DepDelay": "mean"})
data.head()
sns.lineplot(data["Time"], data["DepDelay"])
sns.lineplot(data["Time"], data["ArrDelay"])
plt.show()

sns.lineplot(data = data)
plt.show()

sns.lineplot(x="Time", y="ArrDelay", hue="Origin", data=df[df.Origin.isin(["ATL", "HOU", "IND"])])
plt.show()
```

### conclusions
- hue = useful for draw several lines with different colors
- lineplot, (x and y as names) and pass data for extract both values
- datetime.datetime
- groupby as_index=False, to generate a new index

# plotting several plots using sns
## plotting distribution of one variable
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5),tight_layout=True, sharey=True)
df  =  pd.read_csv("base_datos_2008.csv")
df.dropna(inplace=True, subset = ["ArrDelay", "DepDelay", "Distance"])

sns.distplot(df["Distance"], ax=ax1)
sns.distplot(df["ArrDelay"], ax=ax2)
plt.show()
```
### conclusions
- sns.displot(ax=axis)
   - you should pass ax as axis to plot the figure


# plotting several plots
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4.5),tight_layout=True, sharey=True)
df  =  pd.read_csv("base_datos_2008.csv").head(500)
df.dropna(inplace=True, subset = ["ArrDelay", "DepDelay", "Distance"])
ax1.scatter(x=df.index, y=df["ArrDelay"])
ax2.scatter(x=df.index, y=df["DepDelay"])
ax3.scatter(x=df.index, y=df["Distance"])
plt.show()
```
### conclusions
- axis comes from plt, then you could use any of these methods

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5),tight_layout=True, sharey=True)
df  =  pd.read_csv("base_datos_2008.csv")
df.dropna(inplace=True, subset = ["ArrDelay", "DepDelay", "Distance"])

sns.distplot(df["Distance"], ax=ax1)
sns.distplot(df["ArrDelay"], ax=ax2)
plt.show()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df  =  pd.read_csv("base_datos_2008.csv")
sns.kdeplot(df["ArrDelay"])
sns.kdeplot(df["DepDelay"])
plt.show()
```

# boxplot with sns
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df  =  pd.read_csv("base_datos_2008.csv")
df2 = df[df.Origin.isin(["ATL", "HOU", "IND"])].sample(frac=1)
sns.boxplot(x="DepDelay", y = "Origin", data = df2)
plt.show()
```
### conclusions
x = data
y = subdivision

# scatter plot with sns
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df  =  pd.read_csv("base_datos_2008.csv")
df.dropna(inplace=True, subset = ["ArrDelay", "DepDelay", "Distance", "AirTime"])
df2 = df[df.Origin.isin(["ATL", "HOU", "IND"])].sample(frac=1).head(5000)
sns.jointplot(df2["DepDelay"], df2["ArrDelay"])

df3 = df2[np.abs(df2["DepDelay"])<40]
df3 = df3[np.abs(df3["ArrDelay"])<40]

sns.jointplot(df3["DepDelay"], df3["ArrDelay"], kind="hex")
sns.jointplot(df3["DepDelay"], df3["ArrDelay"], kind="kde")

plt.show()
```

### conclusions
- scatter plot with histograms for each variable
- hive plot and histograms for each variable
- it could be hive or difussion

# heatmap con sns
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df  =  pd.read_csv("base_datos_2008.csv")
pivoted_df = pd.pivot_table(data = df, index = "Month", columns="Origin", values="Distance")
sns.heatmap(pivoted_df, cmap="coolwarm")
plt.show()
```
### conclusions
- good way to visualize three variables in a two dimensional chart
- good way to visualize geographical points

# machine learning
![](img/supervisado_no_supervisado.png)  
## supervisado
modela, se tiene Y, describe una relacion
## no supervisado
clusteriza, no se tiene Y, algunos son generativos

# problema supervisado
- 3 productos, informacion de clientes y productos que se le ofrece, tratar de predecir el mejor producto
basado en la info del cliente
   - encuentra modelos explicativos

# problema no supervisado
- tenemos miles de clientes y necesitamos definir 3 nuevos productos
   - encuentra grupos similares

# preprocesssing

```python
from sklearn import preprocessing
import pandas as pd
import numpy as np
df  =  pd.read_csv("base_datos_2008.csv")
df = df[["ArrDelay", "DepDelay", "Distance", "AirTime"]].dropna()
df.head()
x_scaled = preprocessing.scale(df)
x_scaled.mean(axis=0)
```
### conclusions
normalize the whole dataset using normal distribution or z-score

```python
from sklearn import preprocessing
import pandas as pd
import numpy as np
df = pd.read_csv("base_datos_2008.csv")
df = df[["ArrDelay", "DepDelay", "Distance", "AirTime"]].dropna()
df.dropna(inplace=True)
min_max_scaler = preprocessing.MinMaxScaler([0,10])
x_train = min_max_scaler.fit_transform(df)
x_train.mean(axis = 0)
```
### conclusions
- make sure all your variables are numerical variables
- select a min, max and all your values will be between those values, keeping the relationship

```python
from sklearn import preprocessing
import pandas as pd
import numpy as np
df = pd.read_csv("base_datos_2008.csv")
df = df.dropna(subset=["Origin"])
dummies = pd.get_dummies(df, columns=["Origin"])
dummies.head()
```

# k-means
![](img/80171.png)  
- split data in n groups, each groups with the same variance.
- los nuevos puntos se asignan al centroide mas cercano

# k-means errors
![](img/kmeans_errors.png)  
you could run in troubles in the following situations:
1. wrong number of clusters.
2. there is not way to separate the data with the same variance.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows = 10000)
def train_kmeans(df, columns, clusters):
   df = df[columns].dropna()
   x_train, x_test = train_test_split(df[columns], random_state=0, test_size=0.2)
   model_kmeans = KMeans(n_clusters=clusters, random_state=0)
   model_kmeans.fit(x_train)
   model_kmeans.cluster_centers_
   predictions = model_kmeans.predict(x_test)
   label_counts = np.unique(model_kmeans.labels_, return_counts=True)
   print(f"label counts {label_counts}")
   print(f"labels : {model_kmeans.labels_}")
   print(f"predictions : {predictions[:50]}")
   plt.scatter(x_train[columns[0]], x_train[columns[1]], c=model_kmeans.labels_)
   plt.show()  

columns = ["AirTime", "Distance", "TaxiOut", "ArrDelay", "DepDelay"]
train_kmeans(df, columns, clusters=2)
train_kmeans(df, columns, clusters=3)
train_kmeans(df, columns, clusters=4)
train_kmeans(df, columns, clusters=5)
```
![](img/Figure_1.png)  

### conclusions
- all features to train, needs to be numeric
- break data into train and test set

1. kmeans model creation
   - select cluster number
      - you could train several times with different cluster numbers
   - select random state
      - is for keep the same train out for each run

2. kmeans model fit
   - fit the model with the data

3. kmeans model predict
   - predict the cluster for each data

# hierarchical clustering
calculate distances (euclidean distance) in a hierarchical way
1. between records
2. between records and groups
3. between groups
![](img/hirarchical_clustering.png)  
![](img/88672png)  

```python
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv("base_datos_2008.csv", nrows = 10000)
columns = ["AirTime", "Distance", "TaxiOut", "ArrDelay", "DepDelay"]

df = df[columns].dropna()

x_train, x_test = train_test_split(df[columns], random_state=0, test_size=0.2)

model_hierarchical_cluster = AgglomerativeClustering(
   n_clusters = 5,
   affinity = "euclidean",
   linkage = "ward",
   compute_full_tree = "auto"
)
model_hierarchical_cluster.fit(x_train)
predictions = model_hierarchical_cluster.fit_predict(x_train)

print(f"labels : {model_hierarchical_cluster.labels_}")
print(f"predictions : {predictions[:50]}")

plt.scatter(x_train[columns[0]], x_train[columns[1]], c=predictions)
plt.show()  

predictions_test = model_hierarchical_cluster.fit_predict(x_test)
predictions_test
```
![](img/Figure_2.png)  
### conclusions
- ram consumption is a fact to take care
- it generates a n*n matrix to calculate each distance

# linear regression
- the mean error tendency -> 0
- the idea is minimize the ECM
![](img/regresion_lineal.png)  
  
```python
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv")
df = df.dropna(subset = ["DepDelay", "ArrDelay"])

x_column = ["DepDelay"]
y_column = ["ArrDelay"]

X = df[x_column]
Y = df[y_column]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

model_regression = linear_model.LinearRegression()
model_regression.fit(x_train,y_train)

print(f"coefficients : {model_regression.coef_}")
y_pred = model_regression.predict(x_test)

print("R squared: %.2f" % r2_score(y_test, y_pred))

plt.scatter(X[1:10000], Y[1:10000], color='black')
plt.plot(X[1:10000], model_regression.predict(X[1:10000]), color="blue")
plt.show()
```

### conclusions
- r2_score is use to check how accurate the model is

# linear regression for categorical variables
```python
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows=100000)

x_column = ["DepDelay"]
y_column = ["ArrDelay"]

df = df.dropna(subset=["AirTime", "Distance", "TaxiIn", "TaxiOut"])
X = df[["AirTime", "Distance", "TaxiIn", "TaxiOut"]]
Y = df[y_column]

df["Month"] = df["Month"].apply(str)
df["DayofMonth"] = df["DayofMonth"].apply(str)
df["DayOfWeek"] = df["DayOfWeek"].apply(str)

dummies = pd.get_dummies(data=df[["Month", "DayofMonth", "DayOfWeek", "Origin", "Dest"]])

X = dummies.add(X, fill_value=0)
X.columns.values
X = X.add(df[["DepDelay"]], fill_value=0)

model_regression_categorical = linear_model.LinearRegression()
model_regression_categorical.fit(X,Y)
y_pred = model_regression_categorical.predict(X)
print(f"coefficients : {model_regression_categorical.coef_}")
print("R squared: %.2f" % r2_score(Y, y_pred))
```

### conclusions
- include numerical variables as well to explain most of the data

# logistic regression
![](img/logistic.png)  
the classification could be done in N categories

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
df = pd.read_csv("base_datos_2008.csv", nrows=100000)
df = df.dropna(subset=["ArrDelay"])
df = df.sample(frac=1)
X = df[["DepDelay"]]
Y = df["ArrDelay"] < 30
model_logistic_regression = LogisticRegression()
model_logistic_regression.fit(X,Y)
y_pred = model_logistic_regression.predict(X)

np.round(model_logistic_regression.predict_proba(X), 3)
np.mean(y_pred == Y)
np.mean(Y)

model_confusion_matrix = confusion_matrix(Y, y_pred)
print(model_confusion_matrix)
```
![](img/59721.png)  

### conclusions  
- the main diagonal contains the correct answer, positive and negative
- you need to create a categorical or boolean variable for Y

# Naive-bayes
assumptions: 
- each variable is independent of the others

![](img/naive.png)  
# BernoulliNB

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
df = pd.read_csv("base_datos_2008.csv", nrows=100000)
df = df.sample(frac=1)
df = df.dropna(subset=["ArrDelay"])
Y = df["ArrDelay"] > 0

df["Month"] = df["Month"].apply(str)
df["DayofMonth"] = df["DayofMonth"].apply(str)
df["DayOfWeek"] = df["DayOfWeek"].apply(str)
df["TailNum"] = df["TailNum"].apply(str)
x = pd.get_dummies(data=df[["Month", "DayofMonth", "DayOfWeek", "TailNum", "Origin", "Dest", "UniqueCarrier"]])

clf = BernoulliNB()
clf.fit(x, Y)
y_pred = clf.predict(x)
np.mean(Y == y_pred)
```
### conclusions
- useful for categorical variables

# Gaussian naive bayes
```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
df = pd.read_csv("base_datos_2008.csv", nrows=100000)
df = df.sample(frac=1)
df = df.dropna(subset=["ArrDelay"])
Y = df["ArrDelay"] > 0
X = df[["AirTime", "Distance", "TaxiIn", "TaxiOut"]]
clf = GaussianNB()
clf.fit(X, Y)
y_pred = clf.predict(X)
np.mean(Y == y_pred)
```
### conclusions
- useful for numerical input variables
- for both cases the Y variable must be categorical or boolean

# Decision tree
![](img/84381.png)  

# DecisionTreeClassifier
is useful for categorical, similar to naive models, the Y variable should be categorical or boolean

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

df = pd.read_csv("base_datos_2008.csv")
df = df.sample(frac=1)
df = df.dropna(subset=["ArrDelay"])
Y = df["ArrDelay"] > 10

x_train, x_test, y_train, y_test = train_test_split(df[columns], Y, random_state=0, test_size=0.2)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

Y_pred = clf.predict(x_train)
np.mean(y_train == Y_pred)

Y_pred_test = clf.predict(x_test)
np.mean(y_test  == Y_pred_test)
```
### conclusions  
- useful when "Y" is a categorical value, X could be numerical

# DecisionTreeRegressor
```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

df = pd.read_csv("base_datos_2008.csv")
df = df.sample(frac=1)
df = df.dropna(subset=["ArrDelay"])
Y = df["ArrDelay"]

x_train, x_test, y_train, y_test = train_test_split(df[columns], Y, random_state=0, test_size=0.2)

clf = DecisionTreeRegressor()
clf = clf.fit(x_train, y_train)

Y_pred = clf.predict(x_train)
np.mean(y_train == Y_pred)

Y_pred_test = clf.predict(x_test)
np.mean(y_test  == Y_pred_test)
```
- useful when "Y" is a numerical value, X could be numerical

# Random forest
there are two types of random forest:
- regression: when Y is numeric
- classification: when Y is categorical

both of them let you work with numerical X

# RandomForestClassifier
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows = 100000)
df = df.dropna(subset=["ArrDelay"])
df = df.sample(frac=1)

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

X = df[x_columns]
Y = df["ArrDelay"] > 10

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, Y)
y_pred_test = clf.predict(x_test)
np.mean(y_test == y_pred_test)

print(f"feature_importances {clf.feature_importances_}")
```

# RandomForestRegressor
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows = 100000)
df = df.dropna(subset=["ArrDelay"])
df = df.sample(frac=1)

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

X = df[x_columns]
Y = df["ArrDelay"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

clf = RandomForestRegressor(n_estimators=100)
clf.fit(X, Y)
y_pred_test = clf.predict(x_test)
np.mean(y_test == y_pred_test)

print(f"feature_importances {clf.feature_importances_}")
```

# Support vector machine

# support vector machine classifier
```python
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows = 1000)
df = df.dropna(subset=["ArrDelay"])
df = df.sample(frac=1)

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

X = df[x_columns]
Y = df["ArrDelay"] > 10

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

clf = SVC()
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)

clf = SVC(kernel="linear")
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)

clf = SVC(kernel="sigmoid")
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)
```

# support vector machine regressor
```python
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("base_datos_2008.csv", nrows = 1000)
df = df.dropna(subset=["ArrDelay"])
df = df.sample(frac=1)

x_columns = ["Distance", "AirTime", "DepTime", "TaxiIn", "TaxiOut", "DepDelay"]

X = df[x_columns]
Y = df["ArrDelay"] 

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

clf = SVR()
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)

clf = SVR(kernel="linear")
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)

clf = SVR(kernel="poly")
clf.fit(X, Y)
y_pred = clf.predict(x_test)
np.mean(y_test == y_pred)
```


# knn
![](img/79965.png)  
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("base_datos_2008.csv", nrows = 100000)
df = df[["AirTime", "Distance", "TaxiOut", "ArrDelay"]].dropna()
df["ArrDelay"] = df["ArrDelay"].apply(lambda x: "Delayed" if x > 10 else "Not Delayed") 

X = df[["AirTime", "Distance", "TaxiOut"]]
Y = df["ArrDelay"]

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X, Y)
y_predict  = model_knn.predict(cols)

np.mean(y_predict == Y)
np.mean(Y == "Not Delayed")

from sklearn.metrics import confusion_matrix
confusion_matrix  = confusion_matrix(Y, predictions)
confusion_matrix 
```
### conclusions
1. calculate distance between the new point and all the points
2. select the k nearest points
3. assign the new point to the majority group

# Spark
![](img/23896.png)  

# PySpark

# Map reduce
- lineal scalability 
- write to disk map-reduce results

# DAG (Directed Acyclic Graph)
# resilient databases

```python
docker run -it --name pyspark apache/spark-py:v3.3.0 /opt/spark/bin/pyspark
# from pyspark import SparkConf, SparkContext
# conf = SparkConf().setMaster("local").setAppName("My App")
# sc = SparkContext(conf=conf)
lines = sc.textFile("test.txt")
lines
lines.count()
lines.first()

lines2 = lines.sample(fraction = 0.1, withReplacement = False)
lines2.first()
lines2.count()
```
### conclusions
- run with docker to make sure that the spark is running

# RDD (Resilient Distributed Database)
1. actions (immutable)
2. lazy operation
3. GAD optimized
4. back to states
5. desired RDD

# spark operation
use spark filter to filter the data

```python
result = lines.filter(lambda line: "1" in line)
result
result.count()
result.take(3)

import re
lines.filter(lambda x: re.search(r"[3-9]", x)).count()

numbers = lines.filter(lambda x: "1" in x)
numbers.persist()
```
### conclusions
- use persist to save the operation

```shell
docker cp base_datos_2008.csv pyspark:/opt/spark/work-dir
docker exec -it -u 0 pyspark bash
python3 -m pip install pandas
```

# Spark: reading csv files
- you don't use pandas in this case
- you need to use an sqlContext

```python
from pyspark.sql.types import StringType
from pyspark import SQLContext
import pandas as pd
sqlContext = SQLContext(sc)

dfspark = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("base_datos_2008.csv")


dfspark.show(2)
dfspark.head(2)
dfspark.count()

dfspark = dfspark.sample(fraction=00.1, withReplacement=False)
dfspark.count()

dfspark = dfspark.withColumn("ArrDelay", dfspark["ArrDelay"].cast("integer"))
df2 = dfspark.na.drop(subset = ["ArrDelay", "DepDelay", "Distance"])
df2 = df2.filter("ArrDelay is not NULL")
df2.count()

df2.printSchema()

import numpy as np
media = np.mean(df2.select("ArrDelay").collect())
media
df2.rdd.getNumPartitions()
```
### conclusions
- you could configure the partitions

# spark transformations
```python
from pyspark import SQLContext
import numpy as np
dfspark = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("base_datos_2008.csv")
dfspark = dfspark.sample(fraction=00.1, withReplacement=False)
dfspark = dfspark.withColumn("ArrDelay", dfspark["ArrDelay"].cast("integer"))

df2 = dfspark.na.drop(subset = ["ArrDelay", "DepDelay", "Distance"])
df2 = df2.filter("ArrDelay is not NULL")
df2 = df2.dropDuplicates()

df2.select("ArrDelay").filter("ArrDelay > 60").take(5)

df2.filter("ArrDelay > 100").count()

media = np.mean(df2.select("ArrDelay").collect())
df2.select("ArrDelay").rdd.map(lambda x: (x-media)**2).take(10)
df2.groupBy("DayOfWeek").count().show()

df2.groupBy("DayOfWeek").mean("ArrDelay").show()
df2.select("Origin").rdd.distinct().take(5)

df2.orderBy(df2.ArrDelay.desc()).take(5)
```
### conclusions
- np.f(df.select("Column").collect())
   - useful for summarize some data

# Spark: describe stats by column
```python
dfspark = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("base_datos_2008.csv")
dfspark = dfspark.sample(fraction=00.1, withReplacement=False)
dfspark = dfspark.withColumn("ArrDelay", dfspark["ArrDelay"].cast("integer"))

df2 = dfspark.na.drop(subset = ["ArrDelay", "DepDelay", "Distance"])
df2 = df2.filter("ArrDelay is not NULL")
df2 = df2.dropDuplicates()

df2.select("ArrDelay").describe().show()
df2.select("Origin").rdd.countByValue()

df2.select("ArrDelay").rdd.max()
df2.select("ArrDelay").rdd.collect()

df2.crosstab("DayOfWeek", "Origin").take(2)
```
- in general you could use describe method to get all stats for column, you  need to combine this with a select
- to find max, min and so on, you kneed to apply an rdd first

# Spark: numeric operations
```python
dfspark = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("base_datos_2008.csv")
dfspark = dfspark.sample(fraction=00.1, withReplacement=False)
dfspark = dfspark.withColumn("ArrDelay", dfspark["ArrDelay"].cast("integer"))

df2 = dfspark.na.drop(subset = ["ArrDelay", "DepDelay", "Distance"])
df2 = df2.filter("ArrDelay is not NULL")
df2 = df2.dropDuplicates()

lista = sc.parallelize(range(1, 1000000))
lista.reduce(lambda x,y: x+y)

from pyspark.sql.functions import mean, stddev, col
media = df2.select(mean(col("ArrDelay"))).collect()
std = df2.select(stddev(col("ArrDelay"))).collect()[0][0]

df2.withColumn("Diferencia", df2["ArrDelay"] - df2["DepDelay"]).collect()
```
- you can parallelize function over lot of data using sc.parallelize

# Spark: join
useful to work with spark lists
```python
x = sc.parallelize([("a", 5), ("b", 6), ("c", 7), ("d", 8)])
y = sc.parallelize([("a", 1), ("b", 2), ("c", 3)])
x.join(y).collect()
y.join(x).collect()
y.leftOuterJoin(x).collect()
x.leftOuterJoin(y).collect()
y.rightOuterJoin(x).collect()
```
- you could work with inner, left and right

# Spark: accumulators
```python
lines = sc.textFile("test.txt")
one = sc.accumulator(0)
two = sc.accumulator(0)

def one_counter(line):
   global one, two
   if "1" in line:
      one += 1
      return True
   if "2" in line:
      two += 1
      return True
   else:
      return False

valores=lines.filter(one_counter)
valores

one
two
```

# Spark: map - reduce
map: process a partition
reduce: sum the partition results
```python
dfspark = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("base_datos_2008.csv")
dfspark = dfspark.sample(fraction=00.1, withReplacement=False)
dfspark = dfspark.withColumn("ArrDelay", dfspark["ArrDelay"].cast("integer"))

df2 = dfspark.na.drop(subset = ["ArrDelay", "DepDelay", "Distance"])
df2 = df2.filter("ArrDelay is not NULL")
df2 = df2.dropDuplicates()

A = sc.parallelize(df2.select("Origin").rdd.collect())
A.persist()

mapfunction = A.map(lambda x: (x,1))
mapfunction.collect()

def fun_weight(x):
   if x[0] in ["SEA", "ATL", "HOU"]:
      return ((x, 2))
   elif x[0] == "DEN":
      return ((x, 3))
   else:
      return ((x,1))

A.map(fun_weight).collect()

reduceF = A.map(fun_weight).reduceByKey(lambda x,y: x+y)
reduceF.collect()

reduceF.sortByKey().take(10)
reduceF.sortBy(lambda x:x[1], ascending=False).take(10)

def count_numbers(x):
   if "1" in x: return ("count", (1, 0, 0))
   if "2" in x: return ("count", (0, 1, 0))
   if "3" in x: return ("count", (0, 0, 1))
   return ("count", (0, 0, 0))

mapfunc = lines.map(count_numbers)
mapfunc.reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2])).collect()
```