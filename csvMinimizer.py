import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('Drug1003_analysis/data_0.csv')



# Get the dimensions of the DataFrame (number of rows and columns)
num_rows, num_cols = data.shape

print(f"The CSV file data_0.csv has {num_rows} rows and {num_cols} columns.")
# Select all columns except the first (Assuming the first column is not part of the numerical data)
columns_to_process = data.columns[1:]

# Calculate the quotient of the minimal and maximal values for each column
filtered_columns = []
quotients = {}
i=0
j=0
for column in columns_to_process:
    min_value = data[column].min()
    max_value = data[column].max()
    quotient = min_value / max_value if max_value != 0 else float('inf')  # Handle division by zero
    #quotients[column] = quotient
    if quotient == float('inf'):
        j+=1
    if quotient < 0.00005 and quotient != float('inf'):
        quotients[i] = quotient
        filtered_columns.append(column)
        i+=1


print(j)
print(i)
# Convert the quotients into a DataFrame
quotient_df = pd.DataFrame(quotients, index=[0])

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(quotient_df, bins=100, edgecolor='black', color='skyblue')
plt.xlabel("Quotient")
plt.ylabel("Frequency")
plt.title("Histogram of Quotients of Minimal and Maximal Values")
plt.show()

# Plot the results as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(quotient_df, annot=True, cmap='viridis', fmt=".4f")
plt.title("Quotients of Minimal and Maximal Values (Heatmap)")
plt.xlabel("Columns")
plt.ylabel("Quotient")
plt.show()

# Create a shorter array with only columns having quotients above 0.025
shorter_array = data[filtered_columns].to_numpy()

# Print the shorter array
print(shorter_array)

