import pandas as pd
from learnsk.tree import DecisionTreeClassifier
from learnsk.metrics import accuracy_point

data = pd.read_csv("dataset/Datasset LKS AI Kabupaten Malang 2025.csv")
df = pd.DataFrame(data)

X = df.drop(columns="target")
y = df["target"]

train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

X_train = train_data.drop(columns=["target"]).values
y_train = train_data["target"].values
X_test = test_data.drop(columns=["target"]).values
y_test = test_data["target"].values

model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, min_samples_split=5)
fit_data = model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_point(y_test, pred)

print(f"Accuracy: {accuracy}")