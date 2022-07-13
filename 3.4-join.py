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