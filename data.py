import pandas as pd


df= pd.read_csv(r"F:\Project\DS_Smart_Premium\playground-series-s4e12\train.csv",index_col="id")

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numerical columns with MEDIAN
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with MODE
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

