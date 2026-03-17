from sklearn.datasets import load_wine
import pandas as pd

# load dataset
wine = load_wine()

# create dataframe
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# add target column
df['target'] = wine.target

# save to csv
df.to_csv("data/wine_dataset.csv", index=False)

print("CSV file created: data/wine_dataset.csv")