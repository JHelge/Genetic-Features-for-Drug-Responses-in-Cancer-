
import pandas as pd

# Lade die beiden CSV-Dateien
df1 = pd.read_csv('./Drug1713_analysis/best/features_0.csv', header=None, squeeze=True)
df2 = pd.read_csv('./Drug1714_analysis/best/features_0.csv', header=None, squeeze=True)

# Finde die gemeinsamen Elemente
gemeinsame_elemente = df1[df1.isin(df2)]

# Gib die gemeinsamen Elemente aus
print(gemeinsame_elemente)
