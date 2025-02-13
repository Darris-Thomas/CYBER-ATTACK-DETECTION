import pandas as pd
import glob
import os

# setting the path for joining multiple files
files = os.path.join("dataset", "data*.csv")

# list of merged files returned
files = glob.glob(files)

print("Resultant CSV after joining all CSV files at a particular location...");

# joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
df.to_csv('dataset.csv',index=False)
print(df)