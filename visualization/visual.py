import matplotlib.pyplot as plt
import seaborn as snb
import pandas as pd

data = pd.read_csv("dataset/Datasset LKS AI Kabupaten Malang 2025.csv")
df = pd.DataFrame(data)
y = df["target"]

plt.figure(figsize=(8,5))

plt.pie(y)
plt.title("Decision Tree Classification")
plt.show()