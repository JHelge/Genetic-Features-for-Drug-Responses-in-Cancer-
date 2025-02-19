
import pandas as pd

# Lade die beiden CSV-Dateien
df1 = pd.read_csv('./Drug1378_analysis/best/features_0.csv', header=None)
df2 = pd.read_csv('./Drug1812_analysis/best/features_0.csv', header=None)
print("Drug1378_analysis")
print(df1)
print("Drug1812_analysis")
print(df2)
# Calculate the intersection of the two dataframes
intersection = pd.merge(df1, df2, how='inner')

# Save the intersection to a new CSV file
#intersection.to_csv('intersection.csv', index=False, header=False)

# Optionally, print the intersection
print("Intersection")
print(intersection)
print(intersection.shape)
