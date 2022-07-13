import pandas as pd
import numpy as np
from joblib import Parallel, delayed
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

print(len(df_clean))

if __name__ == '__main__':
   result = Parallel(n_jobs = 2, backend = "multiprocessing")(map(delayed(retraso_maximo), df_clean.values.tolist()))
   print("entro al main con " + str(len(result)))