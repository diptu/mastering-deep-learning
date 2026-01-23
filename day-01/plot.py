import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")

df = pd.read_csv("./data/Housing.csv")

# print(df.head())
sns.scatterplot(x="area", y="price", data=df)
plt.title("Area Vs Price")
plt.show()
