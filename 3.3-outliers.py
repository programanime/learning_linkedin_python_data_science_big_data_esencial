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