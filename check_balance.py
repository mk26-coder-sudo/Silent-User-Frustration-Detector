import pandas as pd

df = pd.read_csv("data/raw/user_behavior.csv")
print(df["frustrated"].value_counts())