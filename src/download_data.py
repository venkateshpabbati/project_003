import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df.to_csv("data/BostonHousing.csv", index=False)
print("Dataset downloaded to data/BostonHousing.csv")