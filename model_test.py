import pandas as pd
from learnsk.tree import DecisionTreeClassifier

data = pd.read_csv("dataset/Datasset LKS AI Kabupaten Malang 2025.csv")
df = pd.DataFrame(data)

features = df.drop(columns=["target"])
labels = df["target"]

# split data 80/20 method
split_data = df.sample(frac=1, random_state=42).reset_index(inplace=True)
split_point = int(0.8 * len(split_data))

model = DecisionTreeClassifier() # on progress
