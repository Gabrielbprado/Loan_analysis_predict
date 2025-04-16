
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/silver/hmeq_silver.csv")

X = df.drop("BAD", axis=1)
y = df["BAD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("../data/gold", exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("../data/gold/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("../data/gold/X_test.csv", index=False)
y_train.to_csv("../data/gold/y_train.csv", index=False)
y_test.to_csv("../data/gold/y_test.csv", index=False)

print("Dados salvos na camada Gold")
